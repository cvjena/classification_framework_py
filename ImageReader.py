import logging

import skimage.io
from Blob import Blob
import PIL.Image
import IAlgorithm
import numpy as np
import os


__author__ = 'simon'


class ImageReader(IAlgorithm.IAlgorithm):
    def __init__(self, mode = 'PIL'):
        self.mode = mode
        assert self.mode in ['PIL', 'OpenCV'], 'Unknown mode passed to ImageReader'

    def _compute(self, blob_generator):
        for blob in blob_generator:
            try:
                # Return image if a path was provided, an skip this layer if data is passed.
                # This is useful for allowing to skip this layer in testing, when you already have the image as nd-array, but
                # don't want to change the architecture
                if blob.data.size == 0 and blob.meta.imagepath:
                    assert os.path.isfile(blob.meta.imagepath) or os.path.islink(blob.meta.imagepath), 'Could not open image %s'%blob.meta.imagepath
                    if self.mode == 'PIL':
                        blob.data = np.float32(PIL.Image.open(blob.meta.imagepath).convert('RGB')) / 255
                    elif self.mode == 'OpenCV':
                        blob.data = np.float32(PIL.Image.open(blob.meta.imagepath).convert('RGB')) / 255
                    yield blob
                else:
                    yield blob
            except OSError as e:
                logging.warning("Could not read image " + blob.meta.imagepath + ": " + str(e))
