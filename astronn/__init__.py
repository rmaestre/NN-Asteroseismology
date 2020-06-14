from abc import ABC, abstractmethod


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


class Model(ABC):
    """
    dataset to load stars preprocessed data.
    """

    @abstractmethod
    def create(self):
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
    def predict(self):
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
