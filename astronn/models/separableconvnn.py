import logging

from astronn import Model

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model


class separableconvnn(Model):
    """
    Main model to work with several channels in asteroseismology
    """

    def create(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        self.model = tf.keras.Sequential(
            [
                layers.SeparableConv1D(
                    kernel_size=10,
                    filters=5,
                    depth_multiplier=3,
                    input_shape=(406, 3),
                    activation="relu",
                ),
                layers.SeparableConv1D(
                    kernel_size=5, filters=5, depth_multiplier=3, activation="relu"
                ),
                layers.SeparableConv1D(
                    kernel_size=2, filters=5, depth_multiplier=3, activation="relu"
                ),
                layers.MaxPool1D(3),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(200, activation="relu"),
                layers.Dense(100, activation="softmax"),
            ]
        )
        self.model.summary()

    def fit(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    def predict(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    def save(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    def load(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass
