import logging

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

import glob
import numpy as np
import pandas as pd

import tensorflow as tf


log = logging.getLogger(__name__)


class precorot:
    """
    Preprocss class to the 77 sample CoRoT stars. We also load an independent estimation of
    logg from Hareter, 2013.
    """

    def __init__(self, conf_file, cols):
        """
        """
        # Loggs and other information is load from the index.csv
        self.conf = pd.read_csv(
            conf_file, header=None, index_col=False, names=cols, sep="\t"
        )

    def preprocess_files(
        self,
        input_folder,
        output_folder,
        num_frequencies=30,
        output_classes=100,
        target="dnu",
    ):
        """
        """
        input_resolution = 0.25
        input_bins = np.arange(-1, 101, input_resolution)

        files = glob.glob(input_folder)
        if len(files) == 0:
            log.warning("Input folder %s is empty!" % input_folder)
        for file in glob.glob(input_folder):
            log.info("Processing frequencies and amplitudes os star %s" % file)
            # read frequency of a given star
            df = pd.read_csv(
                file, header=None, index_col=False, names=["freq", "amp"], sep=" "
            )
            # Check for NaN values
            if df.isnull().values.any():
                log.error("Some malformated value in file %s" % file)
                raise Exception("Some malformated value in file %s" % file)
            # process first N frequencies
            variable_stars = importr("variableStars")
            pandas2ri.activate()
            _res = variable_stars.process(
                frequency=df[["freq"]].values,
                amplitude=df[["amp"]].values,
                filter="uniform",
                gRegimen=0,
                numFrequencies=num_frequencies,
                maxDnu=1,
                minDnu=15,
                dnuGuessError=False,
                debug=False,
                processFirstRangeOnly=num_frequencies,
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
            # get targets based on filename
            file_name = file.split("/")[-1:][0]
            # Info from configuration
            loggs = self.conf[self.conf.corot == int(file_name.split(".")[0])]["loggs"]
            teff = self.conf[self.conf.corot == int(file_name.split(".")[0])]["teff"]
            l = self.conf[self.conf.corot == int(file_name.split(".")[0])]["l"]
            # Stak all channels
            line = np.hstack((dft[0], hd[0], ac[0], loggs, teff, l)).ravel()
            line[pd.isnull(line)] = 0  # NaN to zeros
            line = line[3:]  # drop firsts n values

            # Save to disk
            _df = pd.DataFrame(np.column_stack(line))
            _df.insert(0, "star", file_name.split(".")[0])
            
            _df.to_csv(
                output_folder + file_name.split(".")[0] + ".log",
                index=False,
                header=False,
            )

