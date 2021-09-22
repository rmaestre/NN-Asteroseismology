import glob
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from itertools import product

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

from scipy.signal import find_peaks, peak_widths


def normalize_files(dir):
    for file in glob.glob(dir):
        df = pd.read_csv(file, header=None)
        print(file)
        for index, row in df.iterrows():
            x = row.to_numpy()
            dft = x[1 : 400 + 1]
            hod = x[400 + 1 : (400 * 2) + 1]
            ac = x[(400 * 2) + 1 : (400 * 3) + 1]

            dft[1:80] = 0
            ac[1:80] = 0
            # Get peaks
            peaks, peak_heights = find_peaks(dft, height=0, distance=20)
            peaks_width = peak_widths(dft, peaks, rel_height=0)
            sorted_peaks = [
                x for _, x in sorted(zip(peak_heights["peak_heights"], peaks))
            ]
            
            #dft_norm = np.random.normal(0.1, 0.1, 400)
            dft_norm = np.zeros(400) + 0.1
            value = 1.0
            count = 0
            for peak in sorted_peaks:
                dft_norm[peak] = value
                value /= 0.75
                if count == 5:
                    break
                count +=1

            peaks, peak_heights = find_peaks(ac, height=0, distance=20)
            peaks_width = peak_widths(ac, peaks, rel_height=0)
            sorted_peaks = [
                x for _, x in sorted(zip(peak_heights["peak_heights"], peaks))
            ]
            
            #ac_norm = np.random.normal(0.1, 0.1, 400)
            ac_norm = np.zeros(400) + 0.1
            value = 1.0
            count = 0
            for peak in sorted_peaks:
                ac_norm[peak] = value
                value /= 0.75
                if count == 5:
                    break
                count +=1

            x[1 : 400 + 1] = dft_norm
            x[(400 * 2) + 1 : (400 * 3) + 1] = ac_norm

            if index == 0:
                df_norm = pd.DataFrame(x).T
            else:
                df_norm = df_norm.append(pd.DataFrame(x).T)

        df_norm.to_csv(file + "_norm", index=False, header=False)

#normalize_files("/home/roberto/Downloads/evolutionTracks_line/parts_train/x*")
normalize_files("/home/roberto/Downloads/evolutionTracks_line/parts_validation/x*")
0/0

normalize_files(
    "/home/roberto/Projects/NN-Asteroseismology/astronn/data/deltascuti/preprocessed/*"
)
normalize_files(
    "/home/roberto/Projects/NN-Asteroseismology/astronn/data/bedding/preprocessed/*"
)
normalize_files(
    "/home/roberto/Projects/NN-Asteroseismology/astronn/data/corot/preprocessed/*"
)

"""
rm -rf /home/roberto/Projects/NN-Asteroseismology/astronn/data/deltascuti/preprocessed/*_norm
rm -rf /home/roberto/Projects/NN-Asteroseismology/astronn/data/bedding/preprocessed/*_norm
rm -rf /home/roberto/Projects/NN-Asteroseismology/astronn/data/corot/preprocessed/*_norm
"""