import logging
from Attributes import *

from Blob import Blob


__author__ = 'simon'


class IAlgorithm(object,metaclass=NonOverrideable):

    def __init__(self):
        pass

    #@non_overridable
    def compute(self, in_blob):
        return self._compute(in_blob)

    def _compute(self, blob: Blob):
        logging.info("Using " + str(type(self)) + " compute function.")
        raise NotImplementedError("Not implemented.")

    #@non_overridable
    def compute_all(self, blob_generator):
        return self._compute_all(blob_generator)

    def _compute_all(self, blob_generator):
        logging.info("Using " + str(type(self)) + " compute_all function.")
        for blob in blob_generator:
            yield self.compute(blob)

    #@non_overridable
    def train(self, blob_generator):
        return self._train(blob_generator)

    def _train(self, blob_generator):
        logging.info("Using " + str(type(self)) + " training function.")
        return self.compute_all(blob_generator)

    def requires_training(self):
        logging.info("Training required: " + str(
            getattr(self, "_train").__func__ == getattr(IAlgorithm(), "_train").__func__))
        return getattr(self, "_train").__func__ == getattr(IAlgorithm(), "_train").__func__