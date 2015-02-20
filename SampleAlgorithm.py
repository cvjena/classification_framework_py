
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import vstack

__author__ = 'simon'


class SampleAlgorithm(IAlgorithm):
    def __init__(self):
        pass

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            # Process the blob.data here
            # you can yield as many blobs as you want, just make sure to copy the meta from the input blob to the
            # yielded blobs
            yield blob

    # Optional, only required if your algorithm needs training, otherwise remove this function
    def _train(self, blob_generator):
        # If you need all data at once:
        # Remember the metas!
        # Example
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            data.append(blob.data.ravel())
            labels.append(blob.meta.label)
            metas.append(blob.meta)
        numpy_data = vstack(data)

        # process numpy_data
        # ...

        # Create generator for next layer
        for d,m in zip(data,metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b