import logging

from astronn import Model

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model


class separableconvnn(Model):
    """
    Main model to work with several channels in asteroseismology
    """

    def compile(self, learning_rate):
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
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        self.model.summary()

    def fit(self, dataset, steps_per_epoch, epochs):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """

        history = self.model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)
        return history

    def predict(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    def save(self, path):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        self.model.save(path)

    def load(self, path):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        self.model.load(path)
