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
    args.output_root = args.output_root.replace(".", "_")
    out_root = Path(args.output_root).absolute()
    if world_size > 1:
        out_root = out_root.with_suffix(f".rank{rank}")
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / f"{args.prefix}-fbank80"
    feature_root.mkdir(exist_ok=True)

    id2txt = {}
    sample_ids = []
    prev_wavfile = None
    counter = 0
    wav_counter = -1

    with open(args.wavscp, "r") as f_wav:
        # extract fbank80 feature
        # format example: /mnt/data/librispeech/103-1240-0000.flac CHAPTER ONE ABOUT ...
        # or            : /mnt/data/librispeech/103-1240-0000.flac[13, 45] CHAPTER ONE ABOUT ...
        tempfilename = Path(tempfile.gettempdir()).joinpath(next(tempfile._get_candidate_names())+".wav")
        # tempfilename = str(tempfilename)
        for line in tqdm(f_wav.readlines()):
            # wavfile = line.strip()
            wavfile = line.split()[0].strip()
            m = re.search('^(\S+)\[(\S+),\s*(\S+)\]', wavfile)
            # for format like: test.wav[1.0,3.2]
            if m is not None:
                wavfile = m.group(1)
                stime = float(m.group(2))
                etime = float(m.group(3))
                if etime - stime < 0.2:
                    continue

                if True:
                    if wavfile != prev_wavfile:
                        counter = 0

                        wav_counter += 1
                        if wav_counter % world_size != rank:
                            prev_wavfile = wavfile
                            continue

                        subprocess.run(f"ffmpeg -i {wavfile}  -ar 16000 -f wav -y -loglevel 1 {tempfilename}", shell=True)
                        wav, sample_rate = torchaudio.backend.sox_io_backend.load(tempfilename)
                        # wav, sample_rate = torchaudio.backend.sox_io_backend.load(wavfile, frame_offset=0, num_frames=48000)
                        # print (f"Loading wav file: {wavfile}...")
                        if sample_rate != args.target_sample_rate:
                            wav = F.resample(wav, sample_rate, args.target_sample_rate)
                        # print ("finished resampling")
                        sample_rate = args.target_sample_rate
                    else:
                        if wav_counter % world_size != rank:
                            prev_wavfile = wavfile
                            continue
                        counter += 1
                    sample_id = Path(wavfile).stem + f"_{counter}"
                    prev_wavfile = wavfile
                    ssample = int(stime * args.target_sample_rate)
                    esample = int(etime * args.target_sample_rate)

                    wav_seg = wav[:, ssample:esample]

                    if args.speed != 1.0:
                        assert args.speed > 0.0, "speed pertubation requires speed to be larger than 0"
                        # need to explicitly state the input and output sample rate
                        effects = [["speed", f'{args.speed:.5f}'],
                                   ['rate', f'{args.target_sample_rate}']]
                        wavseg, _ = torchaudio.sox_effects.apply_effects_tensor(
                            wav_seg, args.target_sample_rate, effects,
                        )

                    if args.feat_type == "fbank":
                        extract_fbank_features(wav_seg, args.target_sample_rate, feature_root/f"{sample_id}.npy")
                    elif args.feat_type == "rawwav":
                        # wav_seg: [1, N] -> [N, 1]
                        wav_seg = wav_seg.transpose(0, 1)
                        np.save(feature_root/f"{sample_id}.npy", wav_seg)
                    else:
                        raise NotImplementedError

            else:
                ssample = 0
                nsample = -1
                sample_id = Path(wavfile).stem

                wav, sample_rate = torchaudio.backend.sox_io_backend.load(wavfile)
                if sample_rate != args.target_sample_rate:
                    wav = F.resample(wav, sample_rate, args.target_sample_rate)
                sample_rate = args.target_sample_rate

                if args.feat_type == "fbank":
                    extract_fbank_features(wav, sample_rate, feature_root/f"{sample_id}.npy")
                elif args.feat_type == "rawwav":
                    # wav_seg: [1, N] -> [N, 1]
                    wav = wav.transpose(0, 1)
                    np.save(feature_root/f"{sample_id}.npy", wav)


            sent = ' '.join(line.split()[1:]).strip()
            id2txt[sample_id] = sent
            sample_ids.append(sample_id)

        if args.feat_type == "fbank":
            zip_path = out_root / f"{args.prefix}-fbank80.zip"
        elif args.feat_type == "rawwav":
            zip_path = out_root / f"{args.prefix}-rawwav.zip"
        print ("ZIPing features...")
        create_zip(feature_root, zip_path)
        print ("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(zip_path)

        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for sample_id in sample_ids:
            spk_id = sample_id.split('_')[0]
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(id2txt[sample_id])
            manifest["speaker"].append(spk_id)

        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root/f"{args.prefix}.tsv")


        if world_size > 1:
            print(f"using mpirun with size: {world_size}, not extract global cmvn and generate spm model")
            # args.for_train = False

        # compute global CMVN
        if args.for_train:
            print ("Computing global cmvn stats")
            mean, std = compute_cmvn_stats(zip_path)
            print ("Generating manifest...")

        # Generate vocab
        if args.for_train:
            # if this is for Chinese char: single word for vocabulary, we simply take the most frequent {vocab_size} words
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
        # Clean up
        shutil.rmtree(feature_root)


        # post-processing: collect and merge cmvn
        if args.for_train and  world_size > 1:
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
                    print (f"rank: {r}, root: {feature_root}")
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
                    bpe_type=bpe_type
                )
                # rows = load_tsv_to_dicts(out_root/f"{args.prefix}.tsv")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wavscp", "-w", required=True, type=str)
    parser.add_argument("--for-train", action="store_true")
    parser.add_argument("--prefix", default="train", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char", "chnchar"]),
    parser.add_argument("--feat-type", default="fbank", type=str, choices=["fbank", "rawwav"]),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--target-sample-rate", default=16000, type=int)
    parser.add_argument("--speed", default=1.0, type=float)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
