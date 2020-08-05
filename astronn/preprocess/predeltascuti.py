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


class predeltascuti:
    """
    Preprocss class to the eleven delta scuti stars
    """

    def __init__(self):
        """
        """
        self.targets = {}
        self.targets["kic10661783.lis"] = {"dnu": 39.0, "dr": 7.0}
        self.targets["KIC9851944.lis"] = {"dnu": 26.0, "dr": 5.3}
        self.targets["HD159561.lis"] = {"dnu": 38.0, "dr": 19.0}
        self.targets["CID100866999.lis"] = {"dnu": 56, "dr": np.nan}
        self.targets["HD15082.lis"] = {"dnu": 80.0, "dr": 14.0}
        self.targets["kic4544587.lis"] = {"dnu": 74.0, "dr": 11.0}
        self.targets["KIC8262223.lis"] = {"dnu": 77.0, "dr": 7.10}
        self.targets["HD172189.lis"] = {"dnu": 19.0, "dr": 4.6}
        self.targets["KIC3858884.lis"] = {"dnu": 29.0, "dr": 1.9}
        self.targets["CID105906206.lis"] = {"dnu": 20.0, "dr": 2.61}
        self.targets["KIC10080943.lis"] = {"dnu": 52.0, "dr": 1.7}

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
            log.warning("Input folder %s is empty!" % folder)
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
            # Stak all channels
            line = np.hstack(
                (dft[0], hd[0], ac[0], self.targets[file_name]["dnu"])
            ).ravel()
            line[np.isnan(line)] = 0  # NaN to zeros
            line = line[3:]  # drop firsts n values

            # Save to disk
            _df = pd.DataFrame(np.column_stack(line))
            _df.insert(0, "star", file_name.split(".")[0])
            _df.to_csv(
                output_folder + file_name.split(".")[0] + ".log",
                index=False,
                header=False,
            )

