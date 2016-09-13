from numpy import array
from AlgorithmPipeline import AlgorithmPipeline
from Blob import Blob

from ImageReader import ImageReader
import IAlgorithm
import pyprind


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

    def compute(self, image):
        b = Blob()
        if isinstance(image, str):
            b.meta.imagepath = image
        else:
            b.data = image
        # Use compute all here to incorporate custom compute all functions
        return next(self.pipline.compute([b])).data
    
    def compute_all(self, images):
        in_blobs = list()
        all_images = list(images)
        for im in all_images:
            if isinstance(im, Blob):
                b = im
            else:
                b = Blob()
                if isinstance(im, str):
                    b.meta.imagepath = im
                else:
                    b.data = im
                b.meta.total_num = len(all_images)
            in_blobs.append(b)
        return list([b.data for b in self.pipline.compute(in_blobs)])