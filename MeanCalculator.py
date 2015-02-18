import logging

from numpy import *

import IAlgorithm
from Blob import Blob


__author__ = 'simon'


class MeanCalculator(IAlgorithm.IAlgorithm):
    def __init__(self):
        self.min=Infinity

    def _compute(self, blob):
        logging.info("Calculating mean of image "+blob.meta.imagepath)
        b = Blob()
        b.data = array([mean(blob.data)])
        b.meta = blob.meta
        return b