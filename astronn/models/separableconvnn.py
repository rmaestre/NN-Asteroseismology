import logging
from tensorflow.python.ops.gen_array_ops import Reshape

from astronn import Model

import tensorflow as tf
from tensorflow.keras import layers


class separableconvnn(Model):
    """
    Main model to work with several channels in asteroseismology
    """

    def top_2_categorical_accuracy(self, y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(
            tf.reshape(y_true, (1, y_true.shape[1])), y_pred, k=2
        )

    def compile(self, learning_rate):
        """
        load all files on a given directory (and recursive directories)
        """

        self.model = tf.keras.Sequential(
            [
                layers.GaussianNoise(0.1, input_shape=(400, 2)),
                layers.Convolution1D(
                    kernel_size=20, filters=20, input_shape=(400, 2), activation="elu"
                ),
                layers.MaxPool1D(1),
                layers.Convolution1D(
                    kernel_size=10, filters=10, input_shape=(400, 2), activation="elu"
                ),
                layers.MaxPool1D(1),
                layers.Convolution1D(kernel_size=10, filters=5, activation="elu"),
                layers.MaxPool1D(1),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(100, activation="softmax"),
            ]
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=200, decay_rate=0.9
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        """
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        """

        self.model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        self.model.summary()

    def fit(self, dataset, steps_per_epoch, epochs):
        """
        load all files on a given directory (and recursive directories)
        """

        history = self.model.fit(
            dataset, steps_per_epoch=steps_per_epoch, epochs=epochs
        )
        return history

    def predict_classes(self, data):
        """
        load all files on a given directory (and recursive directories)
        """
        return self.model.predict_classes(data)

    def predict_probs(self, data):
        """
        load all files on a given directory (and recursive directories)
        """
        return self.model.predict_probs(data)

    def save(self, path):
        """
        load all files on a given directory (and recursive directories)
        """
        self.model.save(path)

    def load(self, path):
        """
        load all files on a given directory (and recursive directories)
        """
        self.model.load(path)
