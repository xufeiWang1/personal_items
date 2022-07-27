from examples.speech_recognition_sjtu.data_utils import load_df_from_tsv, save_df_to_tsv
import sys
from pathlib import Path


def is_valid_char(c):
  if c == "|" or c.isdigit() or (c <= 'z' and c >= 'a'):
    return True
  return False

in_tsvfile = sys.argv[1]
out_tsvfile = sys.argv[2]
if len(sys.argv) == 4:
  voc_file = sys.argv[3]
else:
  voc_file = None

char2cnt = dict()

remove_list = []

df = load_df_from_tsv(in_tsvfile)
for i, row in enumerate(df.iterrows()):
  idx, items = row

  if not Path(items["video"]).exists():
      # print (f"deleting {items['video']}")
      # df.drop([idx])
      # continue
      remove_list.append(idx)

  sent = items["tgt_text"].strip().replace(" ", "|").lower().encode("UTF-8").decode("UTF-8")
  sent = ' '.join([w for w in sent if is_valid_char(w)])
  df.at[idx, "tgt_text"] = sent

  for c in sent.split():
    if c in char2cnt:
      char2cnt[c] +=1
    else:
      char2cnt[c] = 1

if len(remove_list) > 0:
    print (f"remove {len(remove_list)} lines ...")
    df.drop(index=remove_list, inplace=True)

save_df_to_tsv(df, out_tsvfile)

sorted_char2cnt = sorted(char2cnt.items(), key=lambda x:x[1], reverse=True)
if voc_file is not None:
  with open(voc_file, 'wt') as f_voc:
    for i, (char, cnt) in enumerate(sorted_char2cnt):
      f_voc.write(f"{char} {cnt}\n")





