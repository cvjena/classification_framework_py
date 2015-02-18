from numpy import array
from Blob import Blob

from ImageReader import ImageReader
import IAlgorithm


__author__ = 'simon'


class Classification(object):
    def __init__(self):
        self.algorithms = [ImageReader()]

    def train(self, dataset):
        in_generator = dataset.blob_generator()
        out_generator = None
        for algo in self.algorithms:
            out_generator = algo.train(in_generator)
            in_generator = out_generator
        # Consume all blobs from the last generator to start the actual calculations
        for b in out_generator:
            pass

    def add_algorithm(self, algorithm):
        self.algorithms.append(algorithm)

    def predict(self, imagepath):
        b = Blob()
        b.meta.imagepath = imagepath
        in_blob = b
        out_blob = array([])
        for algo in self.algorithms:
            out_blob = algo.compute(in_blob)
            in_blob = out_blob
        return out_blob