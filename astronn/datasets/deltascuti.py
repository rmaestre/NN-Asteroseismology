import logging

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

    def parse_csv_line(self, line, n_inputs=1219):
        """
        each file will be parsed with this method. Mainly, we read the
        raw data, split it into three dimensions (vector X) and 
        convert last value of raw data into the expected target
        """
        defs = [tf.constant(np.nan)] * n_inputs
        fields = tf.io.decode_csv(line, record_defaults=defs)
        # Get DFT, HD and AC
        x = tf.stack(tf.split(fields[: 406 * 3], 3), axis=-1)  # Split channels
        # Get Dnu (-1) or dr (-2)
        print(fields[-1:])
        aux = tf.cast(tf.convert_to_tensor(fields[-1:]) / 0.0864, tf.int32)
        y = tf.reshape(tf.one_hot(depth=100, indices=aux, axis=0), (1, 100))
        return x, y

    def csv_reader_dataset(
        self,
        filenames,
        batch_size=32,
        n_parse_threads=5,
        shuffle_buffer_size=10000,
        n_readers=5,
    ):
        """
        """
        dataset = tf.data.Dataset.list_files(filenames)
        dataset = dataset.repeat()
        dataset = dataset.interleave(
            lambda filename: tf.data.TextLineDataset(filename).skip(0),
            cycle_length=n_readers,
        )
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(self.parse_csv_line, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
