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