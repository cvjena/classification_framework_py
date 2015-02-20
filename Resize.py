import logging
from scipy.misc import imresize
from IAlgorithm import IAlgorithm

__author__ = 'simon'


class Resize(IAlgorithm):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = imresize(blob.data,(self.height,self.width))
            yield blob