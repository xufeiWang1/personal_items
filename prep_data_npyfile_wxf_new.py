#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. (c) Facebook, Inc.及其附属公司。
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.#这个源代码是在MIT许可证下授权的，可以在这个源代码树的根目录下的许可证文件中找到。

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
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# world_size = comm.Get_size()
world_size = 1
rank = 0


import pandas as pd
from data_utils_wxf_new import (
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

# /home/chenxie95/gigaspeech-dev/audios/POD1000000004.opus[0.0,10.197]	HEY FRIENDS I DON'T KNOW ABOUT YOU BUT WHEN I WAS A KID I KNEW ABSOLUTELY NOTHING ABOUT MONEY WELL THERE IS A NEW SHOW FROM MARKETPLACE BRAINS ON THAT WILL HELP ADDRESS JUST THAT
# {'id': ['POD1000000004_0', 'POD1000000004_1'], 'audio': [PosixPath('/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80/POD1000000004_0.npy'), PosixPath('/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80/POD1000000004_1.npy')], 'n_frames': [1018, 1247], 'tgt_text': ["HEY FRIENDS I DON'T KNOW ABOUT YOU BUT WHEN I WAS A KID I KNEW ABSOLUTELY NOTHING ABOUT MONEY WELL THERE IS A NEW SHOW FROM MARKETPLACE BRAINS ON THAT WILL HELP ADDRESS JUST THAT", 'MILLIAN BELIZALIAN IS NEW PODCAST FOR KIDS IN THEIR FAMILIES ITS ALL ABOUT MONEY HOW IT WORKS HOW WE USE IT WHERE COMES FROM AND HOW WE COULD ALL GET A LITTLE BIT SMARTER ABOUT IT BUT IN A FUN WAY'], 'speaker': ['POD1000000004', 'POD1000000004']}
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "sentiment", "anger", "disgust", "fear", "happiness", "sadness", "surprise", "tgt_text", "speaker"]

# accumulate the mean and square mean stats to compute mean and invstd 累积平均值和平方平均值统计来计算平均值和invstd
def accum_cmvn_stats(features, mean_sum, square_sum, n_frame):
    if mean_sum is None:
        mean_sum = features.sum(axis=0) # mean_sum 是fbank的按第一维求和，即80个数的array
        square_sum = (features ** 2).sum(axis=0) # 类似，平方求和
        n_frame = features.shape[0] # fbank第一维的个数
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
    args.output_root = args.output_root.replace(".", "_") # 此句没用 ? 将命令行参数output_root /home/xufei/gigaspeech-dev/data4fairseq 中的.换成_
    out_root = Path(args.output_root).absolute() # ? 提取绝对路径 、out_root变成了path类的/home/xufei/gigaspeech-dev/data4fairseq
    if world_size > 1:    # world_size 一直=1就没变过
        out_root = out_root.with_suffix(f".rank{rank}") # 跳过
    out_root.mkdir(exist_ok=True) # ？创建out_root的目录
    # Extract features
    feature_root = out_root / f"{args.prefix}-fbank80" # feature_root是path类的/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80
    feature_root.mkdir(exist_ok=True) # ？ 创建feature_root的目录

    id2txt = {}
    sample_ids = []
    prev_wavfile = None
    counter = 0
    wav_counter = -1

    npy_paths = {}
    npy_lengths = {}
    npy_sentiment = {}
    npy_anger = {}
    npy_disgust = {}
    npy_fear = {}
    npy_happiness = {}
    npy_sadness = {}
    npy_surprise = {}   

    mean_sum, square_sum, n_frame = None, None, 0

    with open(args.wavscp, "r") as f_wav: # 将/home/xufei/gigaspeech-dev/gigaspeech-dev.info打开
        # extract fbank80 feature
        # format example: /mnt/data/librispeech/103-1240-0000.flac CHAPTER ONE ABOUT ...
        # or            : /mnt/data/librispeech/103-1240-0000.flac[13, 45] CHAPTER ONE ABOUT ...
        tempfilename = Path(tempfile.gettempdir()).joinpath(next(tempfile._get_candidate_names())+".wav")# 
        # tempfilename = str(tempfilename)
        for line in tqdm(f_wav.readlines()):# 按行读取gigaspeech-dev.info里的内容
            # wavfile = line.strip()
            wavfile = line.split()[0].strip() # 切分出'/home/chenxie95/gigaspeech-dev/audios/POD1000000004.opus[0.0,10.197]'
            m = re.search('^(\S+)\[(\S+),\s*(\S+)\]', wavfile) # m 是re类，是/home/chenxie95/gigaspeech-dev/audios/POD1000000004.opus[0.0,10.197]的进一步切分
            # for format like: test.wav[1.0,3.2] 
            if m is not None:
                wavfile = m.group(1) # 提取出切片/home/chenxie95/gigaspeech-dev/audios/POD1000000004.opus
                stime = float(m.group(2)) # 提取出切片0.0
                etime = float(m.group(3)) # 提取出切片 10.197
            # for format like: test.wav[1.0,3.2] 
                sentiment = float(line.split()[1].strip())
                anger = float(line.split()[2].strip())
                disgust = float(line.split()[3].strip())
                fear = float(line.split()[4].strip())
                happiness = float(line.split()[5].strip())
                sadness = float(line.split()[6].strip())
                surprise = float(line.split()[7].strip())



                if etime - stime < 0.2: # 如果这句话少于0.2秒 跳过这句话，进行下句话
                    continue

                if True:
                    if wavfile != prev_wavfile: # 当索引文件的n个语句遍历完
                        counter = 0 # 则句子从第0个开始 counter是执行第几个句子

                        wav_counter += 1 # wav_counter也从第0个开始 
                        if wav_counter % world_size != rank:
                            prev_wavfile = wavfile # （是否有过执行？——经过调试发现不会进行
                            continue

                        subprocess.run(f"ffmpeg -i {wavfile}  -ar 16000 -f wav -y -loglevel 1 {tempfilename}", shell=True) # ？ 相当于一个命令行命令
                        wav, sample_rate = torchaudio.backend.sox_io_backend.load(tempfilename) # ？ 
                        # wav, sample_rate = torchaudio.backend.sox_io_backend.load(wavfile, frame_offset=0, num_frames=48000)
                        # print (f"Loading wav file: {wavfile}...")
                        if sample_rate != args.target_sample_rate: # 当采样率不一样时
                            wav = F.resample(wav, sample_rate, args.target_sample_rate) # （是否有过执行？ ——经过调试发现不会进行
                        # print ("finished resampling")
                        sample_rate = args.target_sample_rate # sample_rate是采样率16000
                    else: # 当索引文件的n个语句未遍历完
                        if wav_counter % world_size != rank:
                            prev_wavfile = wavfile
                            continue
                        counter += 1
                    sample_id = Path(wavfile).stem + f"_{counter}" # sample_id是所执行的opus文件_第几句话
                    prev_wavfile = wavfile # 
                    ssample = int(stime * args.target_sample_rate) # ssample是以帧数为单位的采样开始时间
                    esample = int(etime * args.target_sample_rate) # esample是以帧数为单位的采样结束时间

                    wav_seg = wav[:, ssample:esample] # ？ wav_seg是Tensor变量，是语音的以帧数为间隔的波形大小

                    if args.speed != 1.0: # （是否有过执行？ ——经过调试发现不会进行
                        assert args.speed > 0.0, "speed pertubation requires speed to be larger than 0"
                        # need to explicitly state the input and output sample rate 需要显式说明输入和输出采样率
                        effects = [["speed", f'{args.speed:.5f}'],
                                   ['rate', f'{args.target_sample_rate}']]
                        wavseg, _ = torchaudio.sox_effects.apply_effects_tensor(
                            wav_seg, args.target_sample_rate, effects,
                        )

                    if args.feat_type == "fbank":
                        fbank_features = extract_fbank_features(wav_seg, args.target_sample_rate, feature_root/f"{sample_id}.npy") # fbank_features是提取到的一个segment的fbank特征  调用data_utils.py的函数
                        audio_len = fbank_features.shape[0] # audio_len是fbank特征的第一维 fbank是x乘80维的
                        mean_sum, square_sum, n_frame = accum_cmvn_stats(fbank_features, mean_sum, square_sum, n_frame)

                    elif args.feat_type == "rawwav": # （是否会进行？ ——经过调试发现不会进行
                        # wav_seg: [1, N] -> [N, 1]
                        wav_seg = wav_seg.transpose(0, 1)
                        np.save(feature_root/f"{sample_id}.npy", wav_seg)
                        audio_len = wav.shape[0]
                    else:
                        raise NotImplementedError

            else: # （是否会进行? ——经过调试发现不会进行
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
                elif args.feat_type == "rawwav": # （是否会进行？——经过调试发现不会进行
                    # wav_seg: [1, N] -> [N, 1]
                    wav = wav.transpose(0, 1)
                    np.save(feature_root/f"{sample_id}.npy", wav)
                    audio_len = wav.shape[0]

            npy_paths[sample_id] = feature_root/f"{sample_id}.npy" # npy_paths是字典，sample_id=POD1000000004_0，对应 path类/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80/POD1000000004_0.npy
            npy_lengths[sample_id] = audio_len # 同理，npy_lengths是字典，sample_id=POD1000000004_0，对应 1018
            npy_sentiment[sample_id] =  sentiment
            npy_anger[sample_id] = anger
            npy_disgust[sample_id] = disgust
            npy_fear[sample_id] = fear
            npy_happiness[sample_id] = happiness
            npy_sadness[sample_id] = sadness
            npy_surprise[sample_id] = surprise


            sent = ' '.join(line.split()[8:]).strip() # sent 是字符串，字幕
            id2txt[sample_id] = sent # id2txt是字典，sample_id=POD1000000004_0，对应字幕"HEY FRIENDS I DON'T KNOW ABOUT YOU BUT WHEN I WAS A KID I KNEW ABSOLUTELY NOTHING ABOUT MONEY WELL THERE IS A NEW SHOW FROM MARKETPLACE BRAINS ON THAT WILL HELP ADDRESS JUST THAT"
            sample_ids.append(sample_id) # 扩展列表

        """
        if args.feat_type == "fbank":
            zip_path = out_root / f"{args.prefix}-fbank80.zip"
        elif args.feat_type == "rawwav":
            zip_path = out_root / f"{args.prefix}-rawwav.zip"
        print ("ZIPing features...")
        create_zip(feature_root, zip_path)
        print ("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(zip_path)

        """
        # 进度条进行完毕，此时dev-tidy-fbank80文件夹的内容（6748个.npy文件已经写好）
        manifest = {c: [] for c in MANIFEST_COLUMNS} # manifest是字典 内容：{'id': ['POD1000000004_0', 'POD1000000004_1'], 'audio': [PosixPath('/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80/POD1000000004_0.npy'), PosixPath('/home/xufei/gigaspeech-dev/data4fairseq/dev-tidy_xufei-fbank80/POD1000000004_1.npy')], 'n_frames': [1018, 1247], 'tgt_text': ["HEY FRIENDS I DON'T KNOW ABOUT YOU BUT WHEN I WAS A KID I KNEW ABSOLUTELY NOTHING ABOUT MONEY WELL THERE IS A NEW SHOW FROM MARKETPLACE BRAINS ON THAT WILL HELP ADDRESS JUST THAT", 'MILLIAN BELIZALIAN IS NEW PODCAST FOR KIDS IN THEIR FAMILIES ITS ALL ABOUT MONEY HOW IT WORKS HOW WE USE IT WHERE COMES FROM AND HOW WE COULD ALL GET A LITTLE BIT SMARTER ABOUT IT BUT IN A FUN WAY'], 'speaker': ['POD1000000004', 'POD1000000004']}
        for sample_id in sample_ids: # sample_ids是列表
            spk_id = sample_id.split('_')[0] # 切分出POD1000000004等
            manifest["id"].append(sample_id)
            # manifest["audio"].append(audio_paths[sample_id])
            # manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["audio"].append(npy_paths[sample_id])
            manifest["n_frames"].append(npy_lengths[sample_id])
            manifest["tgt_text"].append(id2txt[sample_id])
            manifest["speaker"].append(spk_id)
            manifest["sentiment"].append(npy_sentiment[sample_id])
            manifest["anger"].append(npy_anger[sample_id])
            manifest["disgust"].append(npy_disgust[sample_id])
            manifest["fear"].append(npy_fear[sample_id])
            manifest["happiness"].append(npy_happiness[sample_id])
            manifest["sadness"].append(npy_sadness[sample_id])
            manifest["surprise"].append(npy_surprise[sample_id])
        # 至此字典manifest内容完成，将其存到tsv文件里
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root/f"{args.prefix}.tsv")


        if world_size > 1: # （是否执行？ ——经过调试发现不会进行
            print(f"using mpirun with size: {world_size}, not extract global cmvn and generate spm model") 
            # args.for_train = False

        # compute global CMVN 计算全球CMVN
        if args.for_train: # （是否执行？ ——经过调试发现不会进行
            print ("Computing global cmvn stats ...") 
            # mean, std = compute_cmvn_stats(zip_path)
            save_cmvn_stats(mean_sum, square_sum, n_frame, out_root) # ！！！！ 真正存到npy文件里了的一个操作

        # Generate vocab 生成词汇
        if args.for_train: # （是否执行？ ——经过调试发现不会进行
            # if this is for Chinese char: single word for vocabulary, we simply take the most frequent {vocab_size} words 如果这是中文字符:单个单词作为词汇，我们简单地取最频繁的{vocab_size}单词
            if args.vocab_type == "chnchar":
                spm_filename = f"chnchar_{args.vocab_size}.txt"
                save_chnchar_to_file(manifest["tgt_text"], out_root/spm_filename, args.vocab_size)
                bpe_type = "chnchar"

            else:
                vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
                spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
                spm_filename = f"{spm_filename_prefix}.model"
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

            # Generate config YAML 生成配置YAML
            gen_config_yaml(
                out_root,
                spm_filename=spm_filename,
                specaugment_policy="ld",
                cmvn_type="global",
                gcmvn_path=out_root / "global_cmvn.npy",
                bpe_type=bpe_type
            )
        # Clean up 清除
        # shutil.rmtree(feature_root)


        # post-processing: collect and merge cmvn 后处理:收集并合并CMVN
        if args.for_train and  world_size > 1: # （是否执行？  ——经过调试发现不会进行
            # comm.Barrier()
            # only use rank 0 for merge operation 仅使用秩0进行合并操作
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

                    # collect tsv files 收集tsv文件
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

                # write the final sub-word file 编写最后的子字文件
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

                # Generate config YAML 生成配置YAML
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
