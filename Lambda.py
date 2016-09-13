from IAlgorithm import IAlgorithm

__author__ = 'simon'


class Lambda(IAlgorithm):
    def __init__(self, fun):
        self.fun = fun

    def _compute(self, blob_generator):
        # Design pattern:
        for blob in blob_generator:
            blob.data = self.fun(blob.data)
            yield blob