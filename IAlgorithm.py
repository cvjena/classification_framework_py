from Attributes import *
from Blob import Blob

__author__ = 'simon'


class IAlgorithm(object):
    def __init__(self):
        pass

    @non_overridable
    def compute(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " compute function.")
        for blob in self._compute(blob_generator):
            yield blob

    def _compute(self, blob_generator):
        raise NotImplementedError("Not implemented.")

    @non_overridable
    def train(self, blob_generator):
        return self._train(blob_generator)

    def _train(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " training function.")
        return self.compute(blob_generator)