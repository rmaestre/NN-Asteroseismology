import logging

import glob
import numpy as np
from astronn import Data

import tensorflow as tf


class starmodels(Data):
    """
    This class loads all star models generated with CESTAM and FILOU codes
    """

    def load(self, folder, batch_size=250, add_noise=False):
        """
        method to load all files from a folder
        """
        self.add_noise = add_noise
        return self.csv_reader_dataset(glob.glob(folder), batch_size=batch_size)

    @tf.function
    def is_target_in_range(self, tensor):
        return tf.cond(tensor < 100, True, False)

    def parse_csv_line(self, line, n_inputs=1626):
        """
        each file will be parsed with this method. Mainly, we read the
        raw data, split it into three dimensions (vector X) and 
        convert last value of raw data into the expected target
        """
        defs = [tf.constant(np.nan)] * n_inputs
        fields = tf.io.decode_csv(line, record_defaults=defs)
        # Get DFT, HD and AC
        dft = fields[0:406]
        hod = fields[406 : 406 * 2]
        ac = fields[406 * 2 : 406 * 3]

        # Random noise on random channels [test on training]
        if self.add_noise:
            dft = tf.cond(
                tf.random.uniform([], 0, 1) < 0.5,
                lambda: dft
                + tf.random.normal(
                    shape=tf.shape(dft), mean=dft, stddev=0.2, dtype=tf.float32
                ),
                lambda: tf.convert_to_tensor(dft),
            )

            hod = tf.cond(
                tf.random.uniform([], 0, 1) < 0.5,
                lambda: hod
                + tf.random.normal(
                    shape=tf.shape(hod), mean=hod, stddev=0.2, dtype=tf.float32
                ),
                lambda: tf.convert_to_tensor(hod),
            )

            ac = tf.cond(
                tf.random.uniform([], 0, 1) < 0.5,
                lambda: ac
                + tf.random.normal(
                    shape=tf.shape(ac), mean=ac, stddev=0.2, dtype=tf.float32
                ),
                lambda: tf.convert_to_tensor(ac),
            )

        # Normalizar HoD between 0,1
        hod = tf.math.divide(
            tf.subtract(hod, tf.reduce_min(hod)),
            tf.subtract(tf.reduce_max(hod)*2, tf.reduce_min(hod)),
         )

        x = tf.stack(tf.split(tf.concat([dft, hod, ac], axis=0), 3), axis=-1)
        # Get Dnu (-1) or dr (-2)
        aux = tf.cast(tf.convert_to_tensor(fields[-1:]) / 0.0864, tf.int32)
        # Target to one-hot vector
        y = tf.keras.backend.flatten(tf.one_hot(depth=100, indices=aux))
        # If target value > 100, return False as flag, to be filtered
        return x, y, tf.cond(aux < 100, lambda: True, lambda: False)

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
        dataset = dataset.filter(
            lambda x, y, flag: flag
        )  # Filter y_hat targets markes as False
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
