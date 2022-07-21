#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
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
    convert_to_subwords,
    gen_vocab_to_file,
)
import torchaudio
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

# accumulate the mean and square mean stats to compute mean and invstd
def accum_cmvn_stats(features, mean_sum, square_sum, n_frame):
    if mean_sum is None:
        mean_sum = features.sum(axis=0)
        square_sum = (features ** 2).sum(axis=0)
        n_frame = features.shape[0]
    else:
        mean_sum += features.sum(axis=0)
        square_sum += (features ** 2).sum(axis=0)
        n_frame += features.shape[0]
    return mean_sum, square_sum, n_frame

def save_cmvn_stats(mean_sum, square_sum, n_frame, dirname):
    mean = mean_sum / n_frame
    var = square_sum / n_frame - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    mean = mean.astype("float32")
    std  = std.astype("float32")
    # np.savetxt(dirname.joinpath("fbank80.mean"), mean, delimiter="\n")
    # np.savetxt(dirname.joinpath("fbank80.std"),  std,  delimiter="\n")
    with open(dirname.joinpath("cmvn.stats"), "w") as f_stats:
        f_stats.write(f"{n_frame}\n")
        for m in mean_sum:
            f_stats.write(f"{m} ")
        f_stats.write("\n")
        for v in square_sum:
            f_stats.write(f"{v} ")
        f_stats.write("\n")
    global_cmvn = {"mean": mean, "std": std}
    np.save(dirname.joinpath("global_cmvn.npy"), global_cmvn)

def process(args):
    args.output_root = args.output_root.replace(".", "_")
    out_root = Path(args.output_root).absolute()
    if world_size > 1:
        out_root = out_root.with_suffix(f".rank{rank}")
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / f"{args.prefix}-fbank80"
    if feature_root.exists():
        print (f"Error: target feature dir: {feature_root} exists, please double check!")
        raise ValueError
    feature_root.mkdir(exist_ok=True)

    id2txt = {}
    sample_ids = []
    prev_wavfile = None
    counter = 0
    wav_counter = -1

    npy_paths = {}
    npy_lengths = {}

    mean_sum, square_sum, n_frame = None, None, 0

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
                        fbank_features = extract_fbank_features(wav_seg, args.target_sample_rate, feature_root/f"{sample_id}.npy")
                        audio_len = fbank_features.shape[0]
                        mean_sum, square_sum, n_frame = accum_cmvn_stats(fbank_features, mean_sum, square_sum, n_frame)

                    elif args.feat_type == "rawwav":
                        # wav_seg: [1, N] -> [N, 1]
                        wav_seg = wav_seg.transpose(0, 1)
                        np.save(feature_root/f"{sample_id}.npy", wav_seg)
                        audio_len = wav.shape[0]
                    else:
                        raise NotImplementedError

            else:
                wav_counter += 1
                if wav_counter % world_size != rank:
                    continue

                ssample = 0
                nsample = -1
                sample_id = Path(wavfile).stem

                wav, sample_rate = torchaudio.backend.sox_io_backend.load(wavfile)
                if sample_rate != args.target_sample_rate:
                    wav = F.resample(wav, sample_rate, args.target_sample_rate)
                sample_rate = args.target_sample_rate

                if args.feat_type == "fbank":
                    fbank_features = extract_fbank_features(wav, sample_rate, feature_root/f"{sample_id}.npy")
                    audio_len = fbank_features.shape[0]
                    mean_sum, square_sum, n_frame = accum_cmvn_stats(fbank_features, mean_sum, square_sum, n_frame)
                elif args.feat_type == "rawwav":
                    # wav_seg: [1, N] -> [N, 1]
                    wav = wav.transpose(0, 1)
                    np.save(feature_root/f"{sample_id}.npy", wav)
                    audio_len = wav.shape[0]

            npy_paths[sample_id] = feature_root/f"{sample_id}.npy"
            npy_lengths[sample_id] = audio_len


            sent = ' '.join(line.split()[1:]).strip()
            id2txt[sample_id] = sent
            sample_ids.append(sample_id)

        # compute global CMVN
        if args.for_train:
            print ("Computing global cmvn stats ...")
            save_cmvn_stats(mean_sum, square_sum, n_frame, out_root)

        # text processing by converting words to subwords
        id2txt = convert_to_subwords(id2txt, args)
        if args.for_train:
            subword_texts = list(id2txt.values())
            gen_vocab_to_file(subword_texts, out_root/"vocab.txt", args.vocab_size)

        # generate tsv file
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for sample_id in sample_ids:
            spk_id = sample_id.split('_')[0]
            manifest["id"].append(sample_id)
            manifest["audio"].append(npy_paths[sample_id])
            manifest["n_frames"].append(npy_lengths[sample_id])
            manifest["tgt_text"].append(id2txt[sample_id])
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root/f"{args.prefix}.tsv")

        if args.for_train:
            # Generate config YAML
            gen_config_yaml(
                out_root,
                spm_filename="vocab.txt",
                specaugment_policy="ld",
                cmvn_type="global",
                gcmvn_path=out_root / "global_cmvn.npy",
                bpe_type="subword",
                vocab_type=args.vocab_type
            )


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
                    print (f"rank: {r}, root: {local_out_root}")
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
                gen_vocab_to_file(df["tgt_text"], out_root/"vocab.txt", args.vocab_size)

                # Generate config YAML
                gen_config_yaml(
                    out_root,
                    spm_filename="vocab.txt",
                    specaugment_policy="ld",
                    cmvn_type="global",
                    gcmvn_path=out_root / "global_cmvn.npy",
                    bpe_type="subword",
                    vocab_type=args.vocab_type
                )
                # rows = load_tsv_to_dicts(out_root/f"{args.prefix}.tsv")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wavscp", "-w", required=True, type=str)
    parser.add_argument("--for-train", action="store_true")
    parser.add_argument("--prefix", default="train", required=True, type=str)
    # parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char", "chnchar"]),
    parser.add_argument("--vocab-type", default="eng_spm", type=str, choices=["eng_spm", "chn_char", "eng_char"]),
    parser.add_argument("--spm-model", default="", type=str, help="the path of an existed spm model")
    parser.add_argument("--feat-type", default="fbank", type=str, choices=["fbank", "rawwav"]),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--target-sample-rate", default=16000, type=int)
    parser.add_argument("--speed", default=1.0, type=float)
    args = parser.parse_args()

    if rank == 0:
        from examples.speech_recognition_sjtu.utils import cacheCommands
        cacheCommands(sys.argv)

    process(args)


if __name__ == "__main__":
    main()
