from abc import ABC, abstractmethod
import tensorflow as tf

class Data(ABC):
    """
    dataset to load stars preprocessed data.
    """

    @abstractmethod
    def load(self, folder):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    def normalize(self, tensor):
        """
        normalize vector between 0,1
        """
        return tf.math.divide(
            tf.subtract(tensor, tf.reduce_min(tensor)),
            tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)),
        )


class Model(ABC):
    """
    model with generic operations
    """

    @abstractmethod
    def compile(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    @abstractmethod
    def fit(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    @abstractmethod
    def predict_classes(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    @abstractmethod
    def predict_probs(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    @abstractmethod
    def save(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass

    @abstractmethod
    def load(self):
        """
        load all files on a given directory (and recursive directories)

        :return: [description]
        :rtype: PricingObservation
        """
        pass
