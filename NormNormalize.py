from numpy.linalg import linalg
from IAlgorithm import IAlgorithm

__author__ = 'simon'


class NormNormalize(IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            blob.data = blob.data/linalg.norm(blob.data.ravel())
            yield blob