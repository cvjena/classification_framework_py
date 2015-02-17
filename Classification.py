from ImageReader import ImageReader
import IAlgorithm
from numpy import array
__author__ = 'simon'


class Classification(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.algorithms = [ImageReader()]

    def train(self):
        in_generator = iter(self.dataset.imagepaths)
        out_generator = None
        for algo in self.algorithms:
            label_generator = iter(self.dataset.labels)
            out_generator = algo.train(in_generator,label_generator)
            in_generator = out_generator

    def add_algorithm(self, algorithm : IAlgorithm):
        self.algorithms.append(algorithm)

    def predict(self,imagepath):
        in_array = imagepath
        out_array = array([])
        for algo in self.algorithms:
            out_array = algo.compute(in_array)
            in_array = out_array
        return out_array