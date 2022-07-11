import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Union
import random, string
import time

# copy from data_utils in fairseq
def load_tsv_to_dicts(path: Union[str, Path]) -> List[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        rows = [dict(e) for e in reader]
    return rows

def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def generate_rand_string(k, rand_string_set=None):
    if rand_string_set is not None:
        while True:
            rand_string = ''.join(random.choices(string.ascii_lowercase, k=10))
            if rand_string not in rand_string_set:
                rand_string_set.add(rand_string)
                break
    else:
        rand_string = ''.join(random.choices(string.ascii_lowercase, k=10))
    return rand_string


def write_int_to_binfile(f, n):
    f.write(n.to_bytes(4, byteorder="little"))

def read_int_from_binfile(f):
    n_bytes = f.read(4)
    n = int.from_bytes(n_bytes, byteorder="little")
    return n

def write_str_to_binfile(f, string):
    str_len = len(string.encode())
    write_int_to_binfile(f, str_len)
    f.write(string.encode())

def read_str_from_binfile(f):
    str_len = read_int_from_binfile(f)
    if str_len == 0:
        return None
    string = f.read(str_len).decode()
    return string

def write_numpy_to_binfile(f, mat):
    mat_bytes = mat.tobytes(order='C')
    mat_len = len(mat_bytes)
    mat_dim = mat.shape[1]
    write_int_to_binfile(f, mat_len)
    write_int_to_binfile(f, mat_dim)
    f.write(mat_bytes)

def read_numpy_from_binfile(f):
    mat_len = read_int_from_binfile(f)
    mat_dim = read_int_from_binfile(f)
    mat_bytes = f.read(mat_len)
    mat = np.frombuffer(mat_bytes, dtype=np.float32).reshape(-1, mat_dim)
    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npytsvfile", "-i", type=str, help="input tsv files containing npy files")
    parser.add_argument("--outtsvfile", "-o", type=str, help="output tsv files with hdf5 files")
    parser.add_argument("--outdir", "-d", type=str, help="output directory for hdf5 files")
    parser.add_argument("--datatype", "-t", type=str, default="lmdb",  help="either be hdf5, lmdb or bin")
    parser.add_argument("--chunksize", "-n", type=int, default=5000, help="split into n large files")
    args = parser.parse_args()
    assert args.datatype in ["hdf5", "lmdb", "bin"]

    df = load_df_from_tsv(args.npytsvfile)
    rand_string_set = set()

    Path(args.outdir).mkdir(exist_ok=True)

    if args.datatype == "lmdb":
        import lmdb
        rand_string = generate_rand_string(10, rand_string_set)
        out_dir = Path(args.outdir).joinpath(rand_string)
        out_dir.mkdir(exist_ok=True)

        env = lmdb.open(str(out_dir), map_size=1099511627776, readahead=0)
        txn = env.begin(write=True)
    elif args.datatype == "hdf5":
        import h5py
        # rand_string = generate_rand_string(10, rand_string_set)
        stem_filename = Path(args.outtsvfile).stem
        out_filename = Path(args.outdir)/f"{stem_filename}.hdf5"
        file_hdf5 = h5py.File(out_filename, "w")
    elif args.datatype == "bin":
        rand_string = generate_rand_string(10, rand_string_set) + ".bin"
        out_filename = Path(args.outdir).joinpath(rand_string)
        file_bin = open(str(out_filename), "wb")
    else:
        raise NotImplementedError

    start_time = time.time()
    n_line = len(df)
    for i in range(n_line):
        if i > 0 and i % 10000 == 0:
            elapse_time = time.time() - start_time
            print (f"processing the {i}-th (total: {n_line}) npy file, elapse time: {elapse_time:.1f}s...", flush=True)
            start_time = time.time()

        fileid   = df.loc[i]["id"]
        feature  = np.load(df.loc[i]["audio"])
        n_frames = int(df.loc[i]["n_frames"])
        tgt_text = df.loc[i]["tgt_text"]

        assert feature.shape[0] == n_frames

        if args.datatype == "lmdb":
            txn.put(key=fileid.encode(), value=feature)
            # feature = txt.get(fileid.encode())
            # feature = np.frombuffer(feature, dtype="float32").reshape(-1, 80)
            df.at[i, "audio"] = str(out_dir)

            if i > 0 and i % args.chunksize == 0:
                txn.commit()
                env.close()

                rand_string = generate_rand_string(10, rand_string_set)
                out_dir = Path(args.outdir).joinpath(rand_string)
                out_dir.mkdir(exist_ok=True)

                env = lmdb.open(str(out_dir), map_size=1099511627776, readahead=0)
                txn = env.begin(write=True)
        elif args.datatype == "hdf5":
            file_hdf5[fileid] = feature
            df.at[i, "audio"] = str(out_filename)
        # bin
        else:
            write_str_to_binfile(file_bin, fileid)
            write_numpy_to_binfile(file_bin, feature)
            df.at[i, "audio"] = str(out_filename)

            if i > 0 and i % args.chunksize == 0:
                file_bin.close()

                rand_string = generate_rand_string(10, rand_string_set) + ".bin"
                out_filename = Path(args.outdir).joinpath(rand_string)
                file_bin = open(str(out_filename), "wb")

    if args.datatype == "lmdb":
        txn.commit()
        env.close()
    elif args.datatype == "hdf5":
        file_hdf5.close()
    elif args.datatype == "bin":
        file_bin.close()

    save_df_to_tsv(df, args.outtsvfile)





if __name__ == "__main__":
    main()
