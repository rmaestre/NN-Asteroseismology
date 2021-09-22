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
                layers.Convolution1D(
                    kernel_size=10,
                    filters=10,
                    input_shape=(400, 2),
                    activation="elu"
                ),
                layers.MaxPool1D(5),
                layers.Convolution1D(
                    kernel_size=20,
                    filters=20,
                    activation="elu"
                ),
                layers.MaxPool1D(2),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(100, activation="softmax"),
            ]
        )

        """
        self.model = tf.keras.Sequential(
            [
                layers.Dense(400, input_shape=(400, 2), activation="relu"),
                layers.Dense(200, activation="relu"),
                layers.Dense(100, activation="relu"),
                layers.Dense(50, activation="relu"),
                #layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=(400, 2)),
                #layers.Bidirectional(layers.LSTM(10)),
                #layers.LSTM(50, return_sequences=True, input_shape=(400, 2)),
                #layers.LSTM(100),
                #layers.LSTM(50),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(100, activation="softmax"),
            ]
        )
        """
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=200, decay_rate=0.9
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        """
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        """

        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
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
