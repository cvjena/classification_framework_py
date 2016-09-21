from numpy import array
from AlgorithmPipeline import AlgorithmPipeline
from Blob import Blob

from ImageReader import ImageReader
import IAlgorithm
import pyprind


__author__ = 'simon'


class Classification(object):
    def __init__(self):
        self.pipeline = AlgorithmPipeline()
        self.pipeline.add_algorithm(ImageReader())

    def train(self, dataset):
        out_generator = self.pipeline.train(dataset.blob_generator())
        # Consume all blobs from the last generator to start the actual calculations
        for b in out_generator:
            pass

    def add_algorithm(self, algorithm):
        self.pipeline.add_algorithm(algorithm)

    def compute(self, image):
        b = Blob()
        if isinstance(image, str):
            b.meta.imagepath = image
        else:
            b.data = image
        # Use compute all here to incorporate custom compute all functions
        return next(self.pipeline.compute([b])).data
    
    def _gen_inblobs(self, images):
        for im in pyprind.prog_bar(images):
            if isinstance(im, Blob):
                b = im
            else:
                b = Blob()
                if isinstance(im, str):
                    b.meta.imagepath = im
                else:
                    b.data = im
            yield b
    
    def compute_all(self, images):
        return list([b.data for b in self.pipeline.compute(self._gen_inblobs(images))])