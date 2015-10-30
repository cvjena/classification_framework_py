import logging

from numpy import *

import IAlgorithm
from Blob import Blob


__author__ = 'simon'


class MeanCalculator(IAlgorithm.IAlgorithm):
    def __init__(self):
        self.init_threading()

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = blob.data.ravel().mean()
            yield blob