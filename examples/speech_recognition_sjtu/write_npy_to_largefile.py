import numpy as np
import pandas as pd
import h5py
import lmdb
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npytsvfile", "-i", type=str, help="input tsv files containing npy files")
    parser.add_argument("--outtsvfile", "-o", type=str, help="output tsv files with hdf5 files")
    parser.add_argument("--outdir", "-d", type=str, help="output directory for hdf5 files")
    parser.add_argument("--datatype", "-t", type=str, help="either be hdf5 or lmdb")
    args = parser.parse_args()
    assert args.datatype in ["hdf5", "lmdb"]

    df = load_df_from_tsv(args.npytsvfile)
    rand_string_set = set()

    Path(args.outdir).mkdir(exist_ok=True)

    if args.datatype == "lmdb":
        file_hdf5 = None
        env = lmdb.open(args.outdir, map_size=1099511627776, readahead=0)
        txn = env.begin(write=True)
    else:
        # rand_string = generate_rand_string(10, rand_string_set)
        stem_filename = Path(args.outtsvfile).stem
        out_filename = Path(args.outdir)/f"{stem_filename}.hdf5"
        file_hdf5 = h5py.File(out_filename, "w")
        file_lmdb = None

    start_time = time.time()
    n_line = len(df)
    for i in range(n_line):
        if i > 0 and i % 10000 == 0:
            elapse_time = time.time() - start_time
            print (f"processing the {i}-th (total: {n_line}) npy file, elapse time: {elapse_time:.1f}s...")
            start_time = time.time()
            if args.datatype == "lmdb":
                env.sync()


        fileid   = df.loc[i]["id"]
        feature  = np.load(df.loc[i]["audio"])
        n_frames = int(df.loc[i]["n_frames"])
        tgt_text = df.loc[i]["tgt_text"]

        assert feature.shape[0] == n_frames

        if args.datatype == "lmdb":
            txn.put(key=fileid.encode(), value=feature)
            # feature = txt.get(fileid.encode())
            # feature = np.frombuffer(feature, dtype="float32").reshape(-1, 80)
            df.at[i, "audio"] = args.outdir
        else:
            file_hdf5[fileid] = feature
            df.at[i, "audio"] = str(out_filename)


    if args.datatype == "lmdb":
        txn.commit()
        env.close()
    else:
        file_hdf5.close()

    save_df_to_tsv(df, args.outtsvfile)





if __name__ == "__main__":
    main()
