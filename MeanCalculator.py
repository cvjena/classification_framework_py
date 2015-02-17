import IAlgorithm
from numpy import *

__author__ = 'simon'


class MeanCalculator(IAlgorithm.IAlgorithm):
    def __init__(self):
        pass

    def compute(self, in_array):
        return mean(in_array)