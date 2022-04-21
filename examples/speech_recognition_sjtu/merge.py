import pandas as pd
import glob
import csv

files = glob.glob("./train-xl-rank*.tsv")
print (f"Merging files: {files}")

df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df = pd.concat([pd.read_csv(fname,
                            sep="\t",
                            header=0,
                            encoding="utf-8",
                            escapechar="\\",
                            quoting=csv.QUOTE_NONE,
                            na_filter=False) for fname in files],
               ignore_index=True)

df.to_csv("train-xl.tsv",
          sep='\t',
          header=True,
          index=False,
          encoding="utf-8",
          escapechar="\\",
          quoting=csv.QUOTE_NONE,)
