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
    compute_cmvn_stats,
    compute_cmvn_stats_from_ziplist,
    get_info_from_tsv,
)
import torchaudio
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    tsvfile = Path(args.tsvfile)
    out_root = Path(tsvfile).parents[0]
    # get zip file list from tsv file
    # also get the text for BPE model training
    print ("Getting zipfile list and text from tsv file")
    zip_path_list, sentences = get_info_from_tsv(tsvfile)

    # compute global CMVN
    print (f"Computing global cmvn stats from zip files: {zip_path_list}")
    # mean, std = compute_cmvn_stats_from_ziplist(zip_path_list)

    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    text_file = out_root.joinpath("temp.txt")
    with open(text_file, 'w') as f:
        for t in sentences:
            f.write(t + "\n")
    gen_vocab(
        Path(text_file),
        out_root / spm_filename_prefix,
        args.vocab_type,
        args.vocab_size,)

    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld",
        cmvn_type="global",
        gcmvn_path=out_root / "global_cmvn.npy")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsvfile", "-t", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char"]),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
