import logging

import skimage.io
from Blob import Blob

import IAlgorithm


__author__ = 'simon'


class ImageReader(IAlgorithm.IAlgorithm):
    def __init__(self):
        pass#self.init_threading()

    def _compute(self, blob_generator):
        for blob in blob_generator:
            try:
                # Return image if a path was provided, an skip this layer if data is passed.
                # This is useful for allowing to skip this layer in testing, when you already have the image as nd-array, but
                # don't want to change the architecture
                if blob.data.size == 0 and blob.meta.imagepath:
                    blob.data = skimage.io.imread(blob.meta.imagepath)
                    yield blob
                else:
                    yield blob
            except OSError as e:
                logging.warning("Could not read image " + blob.meta.imagepath + ": " + str(e))