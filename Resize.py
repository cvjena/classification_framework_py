import logging
from scipy.misc import imresize
from skimage.transform import resize
from IAlgorithm import IAlgorithm

__author__ = 'simon'


class Resize(IAlgorithm):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        #self.init_threading()

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = resize(blob.data,(self.height,self.width),1)
            yield blob