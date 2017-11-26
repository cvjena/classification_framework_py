import logging
from scipy.misc import imresize
from IAlgorithm import IAlgorithm
import numpy as np

__author__ = 'simon'


class Noise(IAlgorithm):
    def __init__(self, type, strength):
        self.type = type
        self.strength = strength

    def _compute(self, blob_generator):
        for blob in blob_generator:
            img = blob.data
            noise_type = self.type
            # upside down flip
            if noise_type=="flipud":
                imgn = img[:,::-1,:]
            # left right flip
            elif noise_type=="fliplr":
                imgn = img[:,:,::-1]
            # rotate the image with 90 degrees
            elif noise_type=="rot90":
                imgn = np.transpose(np.rot90(np.transpose(img,[1, 2, 0])),[2, 0, 1])
            # add gaussian noise
            elif noise_type=="gaussian":
                imgn = img + np.random.randn(*img.shape)*self.strength
            # add salt pepper noise
            elif noise_type=="saltpepper":
                imgn = img.copy()
                imgn[ :, np.random.rand(*img.shape[1:]) > 1-self.strength ] = 0
            else:
                logging.error("Noise type '" + str(self.type) + "' not implemented")
            blob.data = imgn
            yield blob