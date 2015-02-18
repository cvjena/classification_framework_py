from scipy.misc import imresize
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import array, ones, resize

__author__ = 'simon'


class Resize(IAlgorithm):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _compute(self, blob: Blob):
        b = Blob()
        b.data = imresize(blob.data,(self.height,self.width))
        b.meta = blob.meta
        return b