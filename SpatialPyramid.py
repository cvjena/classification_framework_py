
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import array_split, mean, hstack

__author__ = 'simon'


class SpatialPyramid(IAlgorithm):
    def __init__(self, levels=3):
        self.levels = levels

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            features = []
            for level in range(self.levels+1):
                # get x splits
                width_split = array_split(blob.data,2**level,axis=1)
                for column in width_split:
                    tiles = array_split(column,2**level,axis=0)
                    for tile in tiles:
                        features.append(mean(mean(tile,axis=0),axis=0).ravel())
            blob.data = hstack(features)
            yield blob