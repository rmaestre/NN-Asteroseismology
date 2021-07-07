import glob
import os
from re import M
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from itertools import product

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import R, pandas2ri, numpy2ri


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


def calculate_rho_roche(feh, nu_rots, R, M, roche_file):
    """
    """
    # (*)From http://www.astro.princeton.edu/~gk/A403/constants.pdf. See "consultas/others" directory
    # (**) From https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    G = 6.67384e-8  # cm^3/(g*s^2)
    Rsun = 6.955e10  # cm
    Msun = 1.9891e33  # g
    Lsun = 3.839e33  # erg/s
    loggsun = 4.44  # cgs(**)
    UA = 1.496e14  # cm
    # rhosun = 1.4480; #cgs (= g/cm^3)
    rho_sun = 1.409  # Mean density in cgs https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/ (by Haynes et al. 2012)
    Dnusun = 134.8  # muHz (Kjeldsen, Bedding & Christensen-Dalsgaard 2008)
    Mbsun = 4.74  # (*) Bolometric absolute magnitude
    F0sun = 278.0  # muHz (aprox.)
    Dnusun0 = 150.0  # muHz, (150-155) from Andy's model
    Teffsun = 5778  # K
    nu_max_sun = 3050  # muHz (Kjeldsen & Bedding 1995)

    # a,da  = multiplicative factor and uncertainty of the Dnu-rho relation. Guo's values are default
    # b,db  = exponent factor and uncertainty of the Dnu-rho relation. Guo's values are default
    # García Hernández et al. (2017): rho/rhosun = a*(Dnu/Dnusun)^b
    a, da = 1.501, 0.096
    b, db = 2.0373, 0.0405

    solarzx = 0.0245
    ypr = 0.235
    dydz = 2.2

    # We need to solve the problem of 1 digits values of 'feh' instead of 3
    # wrong_feh = np.abs(feh / 10) < 1
    # cefiro1_all.loc[wrong_feh, "feh"] = cefiro1_all.loc[wrong_feh, "feh"] * 10
    if np.abs(feh / 10) < 1:
        feh *= 10

    # First, computation of Z, then Y and X
    Z0 = (1.0 - ypr) / ((1.0 / (solarzx * 10 ** (feh / 100))) + (dydz + 1.0))
    # Y = dydz*Z+ypr
    X0 = 1.0 - ypr - Z0 * (1 + dydz)

    O = 2 * np.pi * nu_rots * 1e-6  # Angular rotation frequency
    Rp = R / (
        1 + O ** 2 * (R * Rsun) ** 3 / (3 * G * M * Msun)
    )  # (23) Pérez-Hernández et al. (1999)

    O_c = np.sqrt(
        8 * G * M * Msun / (3 * Rp * Rsun) ** 3
    )  # (2) Pérez-Hernández et al. (1999). This is an approximation that assumes that Rp does not change with omega
    omega = O / O_c  # Omega/Omega_c (1) Pérez-Hernández et al. (1999)

    Re = (1 + omega ** 2 / 2) * Rp  # (38) Paxton et al. (2019)

    rho_spheroid = (
        M * Msun / (4 / 3 * np.pi * Rp * Re ** 2 * Rsun ** 3)
    )  # Another approximation. This is an spheroid, which differ from a Roche volume at omega > 0.6 (see Paxton et al., 2019)

    rho_ss = (
        M * Msun / (4 / 3 * np.pi * R ** 3 * Rsun ** 3)
    )  # This is the same as cefiro1_all['rho']*vr.rho_sun*4/3*np.pi, because an incorrect calculus of 'rho' in previous versions of filou_file.py

    """
    cefiro1_all = cefiro1_all.rename(
        columns={"F0": "n1l0", "F1": "n2l0"}
    )  # Here we rename colum:q
    ns F0 and F1 for consistency
    """

    # Read Roche model
    df_roche = pd.read_csv(roche_file, sep=",")

    Rmean = (
        np.interp(omega, df_roche["Omega/Omega_K"].values, df_roche["<R>/Req"].values)
        * Re
    )  # Mean radius of a Roche model so that the sphere has the same volume as the Roche model

    """
    Rmean = (
        np.interp(omega, roche_model.Oc, roche_model.Rm_e) * Re
    )  # Mean radius of a Roche model so that the sphere has the same volume as the Roche model
    """

    rho_roche = (
        M * Msun / (4 / 3 * np.pi * Rmean ** 3 * Rsun ** 3)
    )  # Mean density of a Roche model

    return rho_roche


def process_file(
    path,
    output_dir="/home/roberto/Downloads/evolutionTracks_line_rho_roche/",
    roche_file="/home/roberto/Projects/NN-Asteroseismology/astronn/data/roche/roche_radii",
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
        nu_rots = np.nan
        M = np.nan
        R = np.nan
        Teff = np.nan
        with open(path, "r") as infile:
            for id, row in enumerate(infile):
                # Get Luminosity information
                if id == 9:
                    try:
                        chunks = " ".join(row.split()).replace(" ", ",").split(",")
                        M = float(chunks[0])
                        R = float(chunks[1])
                        Lum = float(chunks[2])
                        Teff = float(chunks[3])
                    except:
                        print("Not LUM find %s" % path)
                # Get nu_rots
                if id == 13:
                    try:
                        chunks = " ".join(row.split()).replace(" ", ",").split(",")
                        nu_rots = float(chunks[3])
                    except:
                        print("Not nu_rot find %s" % path)
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

        rho_roche = calculate_rho_roche(1, nu_rots, R, M, roche_file)
        # _df.insert(0, "star", file_name.split(".")[0])
        _df = pd.DataFrame(
            {"Teff": [Teff], "Dnu": [dnu], "rho_roche": [rho_roche]},
            columns=["Teff", "Dnu", "rho_roche"],
        )

        # Get filename
        file_name = path.split("/")[-1:][0]
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

"""
process_file(
    "/home/roberto/Downloads/evolutionTracks/FILOU/VO-m220fe-4a164o0rotjpzt5p7-ad/00489-m220fe-4a164o0rotjpzt5p7-ad.frq"
)
0/0
"""

"""
# Iterative approach only for debug purpose
for file in glob.glob("/home/roberto/Downloads/evolutionTracks/FILOU/*/*.frq"):
     print(file)
     process_file(file)
0/0
"""

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

