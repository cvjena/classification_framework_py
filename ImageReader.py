import logging

import skimage.io as io
from Blob import Blob

import IAlgorithm


__author__ = 'simon'


class ImageReader(IAlgorithm.IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob):
        # Return image if a path was provided, an skip this layer if something else is passed.
        # This is useful for allowing to skip this layer in testing, when you already have the image as nd-array, but
        # don't want to change the architecture
        if blob.meta.imagepath:
            logging.info("Reading image from "+blob.meta.imagepath)
            b = Blob()
            b.data = io.imread(blob.meta.imagepath)
            b.meta = blob.meta
            return b
        else:
            return blob