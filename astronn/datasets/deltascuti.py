import logging
import sys

from astronn import Data

import glob
import numpy as np
import pandas as pd

import tensorflow as tf


class deltascuti(Data):
    """
    This class loads eleven delta scuti stars that have been
    well studied before from an asteroseismology point of
    view. Thus values of large separation and rotational
    splitting are provided.
    """

    def load(self, folder, batch_size):
        """
        method to load all files from a folder
        """
        # Process each file
        return self.csv_reader_dataset(glob.glob(folder), batch_size=batch_size)

    def parse_csv_line(self, line, n_inputs=1202):
        """
        each file will be parsed with this method. Mainly, we read the
        raw data, split it into three dimensions (vector X) and 
        convert last value of raw data into the expected target.
        The star name is also returned
        """
        # (n_inputs - 1) as float
        defs = [tf.constant(np.nan)] * (n_inputs - 1)
        # First field is a string
        defs.insert(0, tf.constant("", dtype=tf.string))
        # Read fields
        fields = tf.io.decode_csv(line, record_defaults=defs)
        # Get DFT, HD and AC
        # In this case, vector stars at position 1 because in position 0 is the starID
        dft = fields[1 : 400 + 1]
        hod = fields[400 + 1 : (400 * 2) + 1]
        ac = fields[(400 * 2) + 1 : (400 * 3) + 1]

        # Normalized HoD between 0,1
        #ac = tf.tensor_scatter_nd_update(ac, [[i] for i in range(30)], np.zeros(30))
        #dft = tf.tensor_scatter_nd_update(dft, [[i] for i in range(30)], np.zeros(30))
        ac = tf.math.divide(
            tf.subtract(ac, tf.reduce_min(ac)),
            tf.subtract(tf.reduce_max(ac), tf.reduce_min(ac)),
        )
        hod = tf.math.divide(
            tf.subtract(hod, tf.reduce_min(hod)),
            tf.subtract(tf.reduce_max(hod), tf.reduce_min(hod)),
        )
        
        x = tf.stack(tf.split(tf.concat([ac, dft], axis=0), 2), axis=-1)
        # Get Dnu (-1) or dr (-2)
        y = tf.reshape(
            tf.one_hot(
                depth=100,
                indices=tf.cast(tf.convert_to_tensor(fields[-1:]), tf.int32),
                axis=0,
            ),
            (1, 100),
        )
        return fields[0], x, y

    def csv_reader_dataset(
        self, filenames, batch_size=32, n_parse_threads=5, n_readers=5,
    ):
        """
        """
        dataset = tf.data.Dataset.list_files(filenames)
        dataset = dataset.repeat()
        dataset = dataset.interleave(
            lambda filename: tf.data.TextLineDataset(filename).skip(0),
            cycle_length=n_readers,
        )
        dataset = dataset.map(self.parse_csv_line, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
