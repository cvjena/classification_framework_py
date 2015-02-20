import logging

from numpy import *

import IAlgorithm
from Blob import Blob


__author__ = 'simon'


class MeanCalculator(IAlgorithm.IAlgorithm):
    def __init__(self):
        self.min=Infinity

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = array([mean(blob.data)])
            yield blob