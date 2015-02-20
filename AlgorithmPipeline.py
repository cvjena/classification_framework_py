from numpy import array
from Blob import Blob

from ImageReader import ImageReader
from IAlgorithm import IAlgorithm


__author__ = 'simon'
class AlgorithmPipeline(IAlgorithm):
    def __init__(self):
        self.algorithms = []

    def _train(self, blob_generator):
        in_generator = blob_generator
        out_generator = blob_generator
        for algo in self.algorithms:
            out_generator = algo.train(in_generator)
            in_generator = out_generator
        return out_generator

    def add_algorithm(self, algorithm):
        self.algorithms.append(algorithm)

    def _compute(self, blob_generator):
        in_generator = blob_generator
        out_generator = blob_generator
        for algo in self.algorithms:
            out_generator = algo.compute(in_generator)
            in_generator = out_generator
        return out_generator