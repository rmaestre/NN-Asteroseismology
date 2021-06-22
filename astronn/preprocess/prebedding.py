import logging

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
import uuid

log = logging.getLogger(__name__)


class prebedding:
    """
    Preprocss class to the Bedding star sample. All frequencies are processed 
    directly from SigSpec files
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
        input_bins = np.arange(0, 100.1, 0.25)

        files = glob.glob(input_folder)
        if len(files) == 0:
            log.warning("Input folder %s is empty!" % input_folder)
        for file in glob.glob(input_folder):
            log.info("Processing frequencies and amplitudes os star %s" % file)
            # read frequency of a given star
            df = pd.read_csv(
                file,
                sep="\s+",
                header=None,
                index_col=False,
                names=["f", "signif", "a", "V4", "V5", "V6", "V7"],
            )

            """
            file_name = file.split("/")[-1:][0]
            _df = pd.DataFrame(df.sort_values(by='a', ascending=False).head(1)[["f","a"]])
            _df.insert(0, "tess", file_name.split(".")[0])
            _df.insert(0, "tic", file.split("/")[-1:][0].split(".")[0].split("-")[2])
            _df.to_csv(
                "/tmp/bedding/" + file.split("/")[-1:][0].split(".")[0].split("-")[2] +".log",
                index=False,
                header=False,
            )
            """

            # Check for NaN values
            if df.isnull().values.any():
                log.error("Some malformated value in file %s" % file)
                raise Exception("Some malformated value in file %s" % file)
            # process first N frequencies
            variable_stars = importr("variableStars")
            pandas2ri.activate()
            _res = variable_stars.process(
                frequency=df[["f"]].values,
                amplitude=df[["a"]].values,
                filter="uniform",
                gRegimen=58,
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
            dnu = self.conf[self.conf.tess == file_name.split(".")[0]]["dnu"] / 0.0864
            # Stak all channels
            line = np.hstack(
                    (
                        np.nan_to_num(np.around(dft[0], 3)),
                        np.nan_to_num(np.around(hd[0], 3)),
                        np.nan_to_num(np.around(ac[0], 3)),
                        np.around(dnu, 3)
                    )
                ).ravel()
            line[pd.isnull(line)] = 0  # NaN to zeros

            # Save to disk
            _df = pd.DataFrame(np.column_stack(line))
            _df.insert(0, "star", file_name.split(".")[0])

            _df.to_csv(
                output_folder + file_name.split(".")[0] + ".log",
                index=False,
                header=False,
            )

