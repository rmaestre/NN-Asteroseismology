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


"""
One all evolution tracks are generated; we use the next commands to generate a splitted
batch of them, in order to be processed by tf.Datasets

# all files to one file 
find . -name "*.log" -exec cat '{}' ';' > allevolution_tracks.out
shuf allevolution_tracks.out > allevolution_tracks_shuf.out
# Split into train and validation
rm allevolution_tracks.out
head -500000  allevolution_tracks_shuf.out > allevolution_tracks_shuf_train.out
tail -6624  allevolution_tracks_shuf.out > allevolution_tracks_shuf_validation.out
rm allevolution_tracks_shuf.out

# Create dir and split big file
mkdir parts_train
mkdir parts_validation
mv allevolution_tracks_shuf_train.out parts_train
mv allevolution_tracks_shuf_validation.out parts_validation

# split big file into multiple ones, using 1000 lines per file
cd parts_train
split -l 1000 allevolution_tracks_shuf_train.out
rm allevolution_tracks_shuf_train.out
cd ..
cd parts_validation
split -l 1000 allevolution_tracks_shuf_validation.out
rm allevolution_tracks_shuf_validation.out

"""


def process_file(
    path,
    output_dir="/home/roberto/Downloads/evolutionTracks_line_murphy/",
    MAX_FREQS_PROCESSED=30,
):
    """
    Process a single frequencies file and save it into
    a single line file with the next format:
    n\tl\tm\tvalue ....
    """
    # if np.random.binomial(1,0.5) == 1:
    if True:
        line_out = []
        dirname_output = path.split("/")[-2]
        filename_ouput = path.split("/")[-1]
        sep = "\t"
        # Save constants
        Lum = np.nan
        Teff = np.nan
        with open(path, "r") as infile:
            for id, row in enumerate(infile):
                # Get Luminosity information
                if id == 9:
                    try:
                        chunks = " ".join(row.split()).replace(" ", ",").split(",")
                        Lum = float(chunks[2])
                        Teff = float(chunks[3])
                    except:
                        print("Not LUM or TEFF find %s" % path)
                # Get frecuencies information
                if id > 24:
                    chunks = " ".join(row.split()).replace(" ", ",").split(",")
                    if len(chunks) == 8:  # Info Star at 8 line
                        n = int(chunks[0])
                        l = int(chunks[1])
                        m = int(chunks[2])
                        freq = float(chunks[3])
                        no = int(chunks[5])
                        freq *= 0.0864  # to muHZ
                        if n >= 0 and n < 10 and l < 3:
                            line_out.append(n)
                            line_out.append(l)
                            line_out.append(m)
                            line_out.append(freq)
                            line_out.append(no)

                        
        # check for Murphy band
        def check_in_edge(L, Teff):   
            try:
                L = np.log(L)                 
                red_edge_L = -0.000811 * Teff + 6.672
                blue_edge_L = -0.001000 * Teff + 10.500
                #print(red_edge_L)
                #print(L)
                #print(blue_edge_L)
                return L > red_edge_L and L < blue_edge_L
            except:
                return False

        def murphy_teff(Teff):
            if Teff > 6000:
                return True
            else:
                return False

        # Check if model star is in Murphy bands and conditions
        if check_in_edge(Lum, Teff) and murphy_teff(Teff):

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
                    "Procesed %d freqs (keeping only %d)"
                    % (len(line_out), MAX_FREQS_PROCESSED)
                )
                # Process
                x = np.asarray(line_out).reshape(-1, 5)
                df = pd.DataFrame(x)
                df.columns = ["n", "l", "m", "freq", "no"]
                # Group frequencies by l and m modes and calculate grouped means
                aux = []
                medians = []
                values = df.query("n>=2 & n<=8").groupby(["l", "m"])["freq"].diff()
                for value in values:
                    if not np.isnan(value):
                        aux.append(value)
                    else:
                        if len(aux) > 0:
                            medians.append(np.median(aux))
                        aux = []
                # calculate dnu
                dnu = np.mean(medians)
                # Convert to one-hot vector
                oh = np.zeros(100)
                oh[int(np.round(dnu))] = 1

                for i in range(1):
                    # Process signals
                    input_resolution = 0.25
                    input_bins = np.arange(0, 100.1, 0.25)

                    variable_stars = importr("variableStars")
                    pandas2ri.activate()

                    def apply_visibilities(row):
                        """
                        """
                        if row["l"] == 0:
                            return np.random.uniform(1, 0.85)
                        elif row["l"] == 1:
                            return np.random.uniform(0.85, 0.7)
                        elif row["l"] == 2:
                            return np.random.uniform(0.7, 0.4)
                        elif row["l"] == 3:
                            return np.random.uniform(0.4, 0.0)

                    x_vis = df.apply(apply_visibilities, axis=1)
                    if "vis" in df.columns:
                        df = df.drop(["vis"], axis=1)  # Remove vis column if exists
                    df = x_vis.to_frame().join(df)
                    df.columns = ["vis", "n", "l", "m", "freq", "no"]
                    df_sorted = df.sort_values("vis", ascending=False).head(30)
                    # Add gaussian noise to all Ls
                    # df_sorted["freq"] = np.random.normal(df_sorted["freq"], 0.1)

                    _res = variable_stars.process(
                        frequency=df_sorted["freq"],
                        amplitude=df_sorted["vis"],
                        filter="uniform",
                        gRegimen=58,
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
                            _res.rx2["fresAmps"]
                            .rx2[str(first_proccesed_freq_name)]
                            .rx2["b"],
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
                            _res.rx2["diffHistogram"].rx2["histogram"].rx2["values"],
                            axis=-1,
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

                    # Get filename
                    file_name = path.split("/")[-1:][0]
                    # Info from configuration
                    # Stak all channels
                    line = np.hstack(
                        (
                            dirname_output + "-" + filename_ouput,
                            np.nan_to_num(np.around(dft[0], 3)),
                            np.nan_to_num(np.around(hd[0], 3)),
                            np.nan_to_num(np.around(ac[0], 3)),
                            np.around(dnu, 3),
                        )
                    ).ravel()

                    # plt.figure()
                    # plt.plot(np.around(ac[0], 3))
                    # plt.plot(np.around(dft[0], 3))
                    # plt.plot(np.around(hd[0], 3))
                    # plt.show()
                    # plt.savefig("drop_"+str(i)+".png")
    
                    # Save to disk
                    _df = pd.DataFrame(np.column_stack(line))
                    # _df.insert(0, "star", file_name.split(".")[0])
                    # Save data to file
                    _df.to_csv(
                        output_dir
                        + dirname_output
                        + "/"
                        + file_name.split(".")[0]
                        + ".log",
                        index=False,
                        header=False,
                        mode="a",
                    )



# Output dir to save all models
filou_folder = "/home/roberto/Downloads/evolutionTracks/FILOU/*"

#process_file(
#    "/home/roberto/Downloads/evolutionTracks/FILOU/VO-m220fe-4a164o0rotjpzt5p7-ad/00489-m220fe-4a164o0rotjpzt5p7-ad.frq"
#)


# Iterative approach only for debug purpose
#for file in glob.glob("/home/roberto/Downloads/evolutionTracks/FILOU/*/*.frq"):
#    print(file)
#    process_file(file)


# Selected models
files_to_be_processed = []
parent_path = "/home/roberto/Downloads/evolutionTracks/FILOU/"
with open(
    "/home/roberto/Projects/asteroseismologyNN/selected_models.csv", "r"
) as infile:
    for id, row in enumerate(infile):
        if id > 0:  # skip header
            file = row.split(",")[-1].replace('"', "").replace("\n", "")
            files_to_be_processed.append(parent_path + file)
            
# Launch threads
with Pool(8) as p:
    p.map(
        # process_file, glob.glob("/home/roberto/Downloads/evolutionTracks/FILOU/*/*.frq")
        process_file,
        files_to_be_processed,
    )

