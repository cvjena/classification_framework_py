from Attributes import *
from Blob import Blob


__author__ = 'simon'


class IAlgorithm(object, metaclass=NonOverrideable):
    def __init__(self):
        pass

    @non_overridable
    def compute(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " compute function.")
        return self._compute(blob_generator)

    def _compute(self, blob_generator):
        raise NotImplementedError("Not implemented.")

    @non_overridable
    def train(self, blob_generator):
        return self._train(blob_generator)

    def _train(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " training function.")
        return self.compute(blob_generator)