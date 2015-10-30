import numpy as np
from numpy.ma import squeeze
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import vstack, split
from numpy import genfromtxt

__author__ = 'simon'


class Colorname(IAlgorithm):
    def __init__(self):
        self.mapping = genfromtxt('data/colormapping.txt', delimiter=',')[:,3:]
        self.init_threading()

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            # Map all RGB values to colorname histograms
            if len(blob.data.shape)==2:
                blob.data = (blob.data.astype(np.uint8)/8).astype(np.uint32)
                r=blob.data
                g=blob.data
                b=blob.data
            else:
                r,g,b=split((blob.data.astype(np.uint8)/8).astype(np.uint32),3,axis=2)
            index = r+32*g+32*32*b
            blob.data = self.mapping[squeeze(index)]
            yield blob