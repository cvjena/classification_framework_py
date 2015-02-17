import IAlgorithm
import skimage.io as io

__author__ = 'simon'


class ImageReader(IAlgorithm.IAlgorithm):
    def __init__(self):
        pass

    def compute(self, in_array):
        # Return image if a path was provided, an skip this layer if something else is passed.
        # This is useful for allowing to skip this layer in testing, when you already have the image as nd-array, but
        # don't want to change the architecture
        if type(in_array) == str:
            return io.imread(in_array)
        else:
            return in_array