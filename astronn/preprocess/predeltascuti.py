import logging

from scipy.stats import binned_statistic

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

import glob
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class predeltascuti:
    """
    Preprocss class to the eleven delta scuti stars
    """

    def preprocess_files(self, input_folder, output_folder, num_frequencies=30):
        """
        """
        input_resolution = 0.25
        input_bins = np.arange(0, 101, input_resolution)

        variable_stars = importr("variableStars")
        pandas2ri.activate()

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
            _res = variable_stars.process(
                df[["freq"]].values,
                df[["amp"]].values,
                "uniform",
                0,
                num_frequencies,
                0,
                0,
                False,
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
                statistic="mean",
                bins=input_bins,
            )
            hd = binned_statistic(
                np.stack(
                    _res.rx2["diffHistogram"].rx2["histogram"].rx2["bins"], axis=-1
                )[0],
                np.stack(
                    _res.rx2["diffHistogram"].rx2["histogram"].rx2["values"], axis=-1
                )[0],
                statistic="mean",
                bins=input_bins,
            )
            ac = binned_statistic(
                np.stack(_res.rx2["crossCorrelation"].rx2["index"], axis=-1)[0],
                np.stack(_res.rx2["crossCorrelation"].rx2["autocorre"], axis=-1)[0],
                statistic="mean",
                bins=input_bins,
            )
            # Stak all channels
            line = np.hstack((dft[0], hd[0], ac[0])).ravel()
            line[np.isnan(line)] = 0  # NaN to zeros
            # get targets
            file_name = file.split("/")[-1:][0]
            # Save to disk
            np.savetxt(
                "%s/%s.log" % (output_folder, file_name.split(".")[0]),
                line,
                delimiter=",",
                newline=" ",
            )
