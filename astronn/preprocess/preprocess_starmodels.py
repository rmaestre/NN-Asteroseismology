import glob
import os
import shutil
import pandas as pd
import numpy as np

from multiprocessing import Pool
from itertools import product

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri


def process_file(
    path,
    output_dir="/home/roberto/Downloads/evolutionTracks_line/",
    MAX_FREQS_PROCESSED=30,
):
    """
    Process a single frequencies file and save it into
    a single line file with the next format:
    n\tl\tm\tvalue ....
    """
    if np.random.binomial(1,0.5) == 1:
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
        # Check if directory exists, and overwrite if it is
        if not os.path.exists(output_dir + dirname_output):
            # shutil.rmtree(output_dir + dirname_output)
            os.makedirs(output_dir + dirname_output)

        if len(line_out) / 4 < MAX_FREQS_PROCESSED:
            print(
                "Not enough frequencies (%d) for star %s"
                % (len(line_out), dirname_output + "-" + filename_ouput)
            )
        else:
            print(
                "Procesed %d freqs (keeping only %d)" % (len(line_out), MAX_FREQS_PROCESSED)
            )
            # Process
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
            # Convert to one-hot vector
            oh = np.zeros(100)
            oh[int(np.round(dnu))] = 1
                
            for i in range(4):
                # Process signals
                input_resolution = 0.25
                input_bins = np.arange(-1, 101, input_resolution)

                variable_stars = importr("variableStars")
                pandas2ri.activate()

                np.random.shuffle(x)
                
                _res = variable_stars.process(
                    frequency=x[:,3],
                    amplitude=np.random.uniform(0,1,len(x)),
                    filter="uniform",
                    gRegimen=0,
                    numFrequencies=30,
                    maxDnu=1,
                    minDnu=15,
                    dnuGuessError=False,
                    debug=False,
                    processFirstRangeOnly=30,
                )
                # Get first group of processed frequencies
                first_proccesed_freq_name = _res.rx2["fresAmps"].names[0]
                # All channel is binned respect the input_resolution
                dft = binned_statistic(
                    np.stack(
                        _res.rx2["fresAmps"]
                        .rx2[str(first_proccesed_freq_name)]
                        .rx2["fInv"],
                        axis=-1,
                    )[0],
                    np.stack(
                        _res.rx2["fresAmps"].rx2[str(first_proccesed_freq_name)].rx2["b"],
                        axis=-1,
                    )[0],
                    statistic="max",
                    bins=input_bins,
                )
                hd = binned_statistic(
                    np.stack(
                        _res.rx2["diffHistogram"].rx2["histogram"].rx2["bins"], axis=-1
                    )[0],
                    np.stack(
                        _res.rx2["diffHistogram"].rx2["histogram"].rx2["values"], axis=-1
                    )[0],
                    statistic="max",
                    bins=input_bins,
                )
                ac = binned_statistic(
                    np.stack(_res.rx2["crossCorrelation"].rx2["index"], axis=-1)[0],
                    np.stack(_res.rx2["crossCorrelation"].rx2["autocorre"], axis=-1)[0],
                    statistic="max",
                    bins=input_bins,
                )

                file_name = path.split("/")[-1:][0]
                # Info from configuration
                # Stak all channels
                line = np.hstack(
                    (
                        np.around(dft[0], 3),
                        np.around(hd[0], 3),
                        np.around(ac[0], 3),
                        np.around(dnu, 3),
                    )
                ).ravel()
                line[pd.isnull(line)] = 0  # NaN to zeros
                line = line[3:]  # drop firsts n values

                # Save to disk
                _df = pd.DataFrame(np.column_stack(line))
                # _df.insert(0, "star", file_name.split(".")[0])
                # Save data to file
                _df.to_csv(
                    output_dir + dirname_output + "/" + file_name.split(".")[0] + ".log",
                    index=False,
                    header=False,
                    mode="a",
                )


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