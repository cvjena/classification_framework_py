
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import array, ones

__author__ = 'simon'


class SampleAlgorithm(IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob: Blob):
        b = Blob()
        b.data = ones(1)
        b.meta = blob.meta
        return b

    # Optional, only makes sense if you want to program a (faster) batch processing variant
    def _compute_all(self, blob_generator):
        # Recommended design pattern:
        # Use yield!
        # for in_array in in_array_generator:
        # yield self.compute(in_array)
        return IAlgorithm._compute_all(self, blob_generator)

    # Optional, only required if your algorithm needs training, otherwise remove this function
    def _train(self, blob_generator):
        # If you need all data at once:
        # Remember the metas!
        # Example
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            data.append(blob.data.flatten())
            labels.append(blob.meta.label)
            metas.append(blob.meta)
        numpy_data = array(data)

        # process data
        # ...

        # Create generator for next layer
        for d,m in zip(data,metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b