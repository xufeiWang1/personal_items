import random
import time
import numpy as np
from pathlib import Path
import glob

from write_npy_to_bin import read_int_from_binfile, read_str_from_binfile, read_numpy_from_binfile

bin_files = glob.glob("/mnt/data/GigaSpeech/debug_npyfiles/bin_files/*bin")

random.shuffle(bin_files)
file_index = 0
num_bin_file = len(bin_files)

bin_filename = bin_files[0]
file_bin = open(bin_filename, "rb")

n_sample = 100000

for i in range(20):
    counter = 0
    start_time = time.time()

    while counter < n_sample:
        try:
            key = read_str_from_binfile(file_bin)
            if key is None:
                raise StopIteration
            value = read_numpy_from_binfile(file_bin)
            counter += 1

        except StopIteration:
            file_bin.close()

            file_index = (file_index + 1 ) % num_bin_file
            bin_filename = bin_files[file_index]
            file_bin = open(bin_filename, "rb")

    end_time = time.time()
    elapse_time = end_time - start_time

    print (f"{i}-th trial: reading {n_sample} samples in {elapse_time:.2f} sec")




