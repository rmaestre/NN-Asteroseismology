import glob
import os
import shutil
import pandas as pd
import numpy as np

from multiprocessing import Pool
from itertools import product

import csv



def process_file(
    path,
    output_file="/tmp/files_dnus.csv"
):
    """
    Process a single frequencies file and save it into
    a single line file with the next format:
    n\tl\tm\tvalue ....
    """
    line_out = []
    dirname_output = path.split("/")[-2]
    filename_ouput = path.split("/")[-1]
    sep = "\t"
    with open(path, "r") as infile:
        for id, row in enumerate(infile):
            if id > 24:
                chunks = " ".join(row.split()).replace(" ", ",").split(",")
                if len(chunks) == 8:  # Info Star at 8 line
                    n = int(chunks[0])
                    l = int(chunks[1])
                    m = int(chunks[2])
                    freq = float(chunks[3])
                    no = int(chunks[5])
                    freq *= 0.0864  # to muHZ
                    if n > 0 and n < 10 and l < 3:
                        line_out.append(n)
                        line_out.append(l)
                        line_out.append(m)
                        line_out.append(freq)
                        line_out.append(no)
    if len(line_out) > 0:
        x = np.asarray(line_out).reshape(-1, 5)
        df = pd.DataFrame(x)
        df.columns = ["n", "l", "m", "freq", "no"]
        # Group frequencies by l and m modes and calculate grouped means
        aux = []
        means = []
        values = df.query('n>2 & n<8').groupby(["l", "m"])["freq"].diff()
        for value in values:
            if not np.isnan(value):
                aux.append(value)
            else:
                if len(aux) > 0:
                    means.append(np.mean(aux))
                aux = []
        # calculate dnu
        dnu = np.median(means)

        # Append to file
        with open(output_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([dirname_output+"/"+filename_ouput, round(dnu, 3)])

# Output dir to save all models
output_dir = "/home/roberto/Downloads/evolutionTracks_line/"
filou_folder = "/home/roberto/Downloads/evolutionTracks/FILOU/*"

# Iterative approach only for debug purpose
#for file in glob.glob("/home/roberto/Downloads/evolutionTracks/FILOU/*/*.frq"):
#   process_file(file)

with Pool(8) as p:
    p.map(
        process_file, glob.glob("/home/roberto/Downloads/evolutionTracks/FILOU/*/*.frq")
    )