import logging

from astronn import Model

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model


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

        :return: [description]
        :rtype: PricingObservation
        """
        self.model = tf.keras.Sequential(
            [
                layers.SeparableConv1D(
                    kernel_size=5,
                    filters=20,
                    depth_multiplier=3,
                    input_shape=(406, 3),
                    activation="selu",
                ),
                layers.MaxPool1D(2),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Flatten(),
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

        history = self.model.fit(
            dataset, steps_per_epoch=steps_per_epoch, epochs=epochs
        )
        return history

    def predict_classes(self, data):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        return self.model.predict_classes(data)

    def predict_probs(self, data):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        return self.model.predict_probs(data)

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
