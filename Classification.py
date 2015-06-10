from numpy import array
from AlgorithmPipeline import AlgorithmPipeline
from Blob import Blob

from ImageReader import ImageReader
import IAlgorithm


__author__ = 'simon'


class Classification(object):
    def __init__(self):
        self.pipline = AlgorithmPipeline()
        self.pipline.add_algorithm(ImageReader())

    def train(self, dataset):
        out_generator = self.pipline.train(dataset.blob_generator())
        # Consume all blobs from the last generator to start the actual calculations
        for b in out_generator:
            pass

    def add_algorithm(self, algorithm):
        self.pipline.add_algorithm(algorithm)

    def predict(self, imagepath):
        b = Blob()
        b.meta.imagepath = imagepath
        # Use compute all here to incorporate custom compute all functions
        return next(self.pipline.compute([b]))

    def compute(self, blob_generator):
        return self.pipline.compute(blob_generator)