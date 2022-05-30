#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import re
import torchaudio.functional as F
import librosa
import numpy as np
import subprocess
import tempfile
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


import pandas as pd
from data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
    load_df_from_tsv,
    compute_cmvn_stats,
    save_chnchar_to_file,
)
import torchaudio
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root).absolute()
    if world_size > 1:
        out_root = out_root.with_suffix(f".rank{rank}")
    out_root.mkdir(exist_ok=True)

    zip_path = out_root / f"{args.prefix}-fbank80.zip"
    if world_size > 1:
        mean, std = compute_cmvn_stats(zip_path)
        local_tsv_filename = out_root.joinpath(f"{args.prefix}.tsv")
        manifest = load_df_from_tsv(local_tsv_filename)
        if args.vocab_type == "chnchar":
            spm_filename = f"chnchar_{args.vocab_size}.txt"
            save_chnchar_to_file(manifest["tgt_text"], out_root/spm_filename, args.vocab_size)
            bpe_type = "chnchar"

        else:
            vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
            spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
            with NamedTemporaryFile(mode="w") as f:
                for t in manifest["tgt_text"]:
                    f.write(t + "\n")
                gen_vocab(
                    Path(f.name),
                    out_root / spm_filename_prefix,
                    args.vocab_type,
                    args.vocab_size,
                )
            bpe_type = "sentencepiece"

        # Generate config YAML
        gen_config_yaml(
            out_root,
            spm_filename=spm_filename,
            specaugment_policy="ld",
            cmvn_type="global",
            gcmvn_path=out_root / "global_cmvn.npy",
            bpe_type=bpe_type
        )

    if True:
        # collect and merge cmvn
        if world_size > 1:
            comm.Barrier()
            # only use rank 0 for merge operation
            if rank == 0:
                out_root = Path(args.output_root).absolute()
                out_root.mkdir(exist_ok=True)
                mean_sum, square_sum = None, None
                n_frame = 0

                df_list = []
                word2id, word2cnt = {}, {}

                for r in range(world_size):
                    local_out_root = out_root.with_suffix(f".rank{r}")
                    local_stats_filename = local_out_root.joinpath("cmvn.stats")
                    with open(local_stats_filename, 'r') as f_stats:
                        # collect global cmvn stats
                        lines = f_stats.read().strip().split('\n')
                        assert len(lines) == 3
                        n_frame += float(lines[0])
                        local_mean_sum = np.array([float(v) for v in lines[1].strip().split()])
                        local_square_sum = np.array([float(v) for v in lines[2].strip().split()])
                        if mean_sum is None:
                            mean_sum = local_mean_sum
                        else:
                            mean_sum += local_mean_sum

                        if square_sum is None:
                            square_sum = local_square_sum
                        else:
                            square_sum += local_square_sum

                    # collect tsv files
                    local_tsv_filename = local_out_root.joinpath(f"{args.prefix}.tsv")
                    local_df = load_df_from_tsv(local_tsv_filename)
                    df_list.append(local_df)

                mean_sum = mean_sum.astype("float32")
                square_sum = square_sum.astype("float32")
                mean = mean_sum / n_frame
                var = square_sum / n_frame - mean ** 2
                std = np.sqrt(np.maximum(var, 1e-8))

                global_cmvn = {"mean": mean, "std": std}
                np.save(out_root.joinpath("global_cmvn.npy"), global_cmvn)

                df = pd.concat(df_list)
                tsv_filename = out_root.joinpath(f"{args.prefix}.tsv")
                save_df_to_tsv(df, tsv_filename)

                # write the final sub-word file
                if args.vocab_type == "chnchar":
                    spm_filename = f"chnchar_{args.vocab_size}.txt"
                    save_chnchar_to_file(df["tgt_text"], out_root/spm_filename, args.vocab_size)
                else:
                    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
                    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
                    with NamedTemporaryFile(mode="w") as f:
                        for t in df["tgt_text"]:
                            f.write(t + "\n")
                        gen_vocab(
                            Path(f.name),
                            out_root / spm_filename_prefix,
                            args.vocab_type,
                            args.vocab_size,
                        )

                # Generate config YAML
                gen_config_yaml(
                    out_root,
                    spm_filename=spm_filename,
                    specaugment_policy="ld",
                    cmvn_type="global",
                    gcmvn_path=out_root / "global_cmvn.npy",
                    bpe_type=args.vocab_type
                )
                # rows = load_tsv_to_dicts(out_root/f"{args.prefix}.tsv")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wavscp", "-w", required=True, type=str)
    parser.add_argument("--for-train", action="store_true")
    parser.add_argument("--prefix", default="train", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char", "chnchar"]),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--target-sample-rate", default=16000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
