from Attributes import *
from Blob import Blob


__author__ = 'simon'


class IAlgorithm(object, metaclass=NonOverrideable):
    def __init__(self):
        use_cache = False
        _cached_blobs = []

    @non_overridable
    def compute(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " compute function.")
        #        if self.use_cache:
        #            if len(self._cached_blobs)>0:
        #                return self._cached_blobs
        #            else:
        #                for blob in self._compute(blob_generator):
        #                    self._cached_blobs.append(blob)
        #                    yield blob
        #        else:
        return self._compute(blob_generator)

    def _compute(self, blob_generator):
        raise NotImplementedError("Not implemented.")

    @non_overridable
    def train(self, blob_generator):
        return self._train(blob_generator)

    def _train(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " training function.")
        return self.compute(blob_generator)