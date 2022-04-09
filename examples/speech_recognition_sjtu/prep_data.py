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

import pandas as pd
from examples.speech_recognition_sjtu.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
import torchaudio
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)

    id2txt = {}
    sample_ids = []

    with open(args.wavscp, "r") as f_wav, open(args.refscp, "r") as f_ref:
        ### read reference text
        # format:
        #      sample_id sentence
        #  e.g.103-1240-0000 CHAPTER ONE ...
        for line in f_ref:
            sample_id = line.split()[0].strip()
            sent = ' '.join(line.split()[1:]).strip()
            id2txt[sample_id] = sent

        # extract fbank80 feature
        # format: /mnt/data/librispeech/103-1240-0000.flac
        for line in tqdm(f_wav.readlines()):
            wavfile = line.strip()
            sample_id = Path(wavfile).stem
            wav, sample_rate = torchaudio.backend.sox_io_backend.load(wavfile)
            extract_fbank_features(wav, sample_rate, feature_root/f"{sample_id}.npy")
            sample_ids.append(sample_id)
            assert sample_id in id2txt, f"Error: wav file with sample_id ({sample_id}) not found in reference text file"

        zip_path = out_root / f"{args.prefix}-fbank80.zip"
        print ("ZIPing features...")
        create_zip(feature_root, zip_path)
        print ("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(zip_path)
        # gen_cmvn_stats(zip_path)
        print ("Generating manifest...")

        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for sample_id in sample_ids:
            spk_id = sample_id.split('_')[0]
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(id2txt[sample_id])
            manifest["speaker"].append(spk_id)

        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root/f"{args.prefix}.tsv")

        # Generate vocab
        if args.for_train:
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

            # Generate config YAML
            gen_config_yaml(
                out_root,
                spm_filename=spm_filename_prefix + ".model",
                specaugment_policy="ld"
            )
        # Clean up
        shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wavscp", "-w", required=True, type=str)
    parser.add_argument("--refscp", "-r", required=True, type=str)
    parser.add_argument("--for-train", action="store_true")
    parser.add_argument("--prefix", default="train", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", required=True, type=str, choices=["bpe", "unigram", "char"]),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
