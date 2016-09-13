from IAlgorithm import IAlgorithm
import numpy as np
import pyprind
import Blob

__author__ = 'simon'


class Concat(IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob_generator):
        bar = None 
        concat = list()
        for blob in blob_generator:
            if bar is None:
                bar = pyprind.ProgBar(blob.meta.total_num)
            concat.append(blob.data.ravel())
            bar.update()
        blob = Blob.Blob()
        blob.data = np.vstack(concat)
        yield blob