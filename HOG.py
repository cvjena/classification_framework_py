from skimage.feature import hog
from skimage import color
from IAlgorithm import IAlgorithm
from Blob import Blob


__author__ = 'simon'


class HOG(IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob: Blob):
        image = color.rgb2gray(blob.data)

        b = Blob()
        b.data = hog(image, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1))
        b.meta = blob.meta
        return b