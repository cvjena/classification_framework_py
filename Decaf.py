import logging
from numpy.core.umath import arctan2, cos, sin
from skimage import color
from IAlgorithm import IAlgorithm
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata
from scipy.misc import imresize


__author__ = 'simon'


class Decaf(IAlgorithm):
    def __init__(self,model,meta,layer):
        self.model = model
        self.meta = meta
        self.net = JeffNet(self.model, self.meta)
        self.layer = layer

    def _compute(self, blob_generator):
        for blob in blob_generator:
            image = blob.data
            # Calculate features
            image = imresize(image,(256,256,3),interp='bicubic')
            score = self.net.classifiyWholeImage(image)
            if self.layer=="score":
                blob.data = (score)
            else:
                blob.data = (self.net.feature(self.layer)[0].reshape((1,-1))[0])
            yield blob