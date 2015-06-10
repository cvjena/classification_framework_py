from numpy import Infinity
from numpy.core.umath import isnan
from numpy.linalg import linalg
from numpy.ma import minimum, maximum, divide
from IAlgorithm import IAlgorithm

__author__ = 'simon'


class MinMaxNormalize(IAlgorithm):
    def __init__(self):
        self.min = None
        self.max = None

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            blob.data = divide(blob.data,self.max)
            yield blob

    def _train(self, blob_generator):
        for blob in blob_generator:
            if self.min is None or self.max is None:
                self.min = blob.data
                self.max = blob.data
            else:
                self.min = minimum(self.min,blob.data)
                self.max = maximum(self.max,blob.data)
            yield blob
        #self.max[isnan(self.min)] = 1