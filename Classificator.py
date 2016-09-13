from sklearn.svm import LinearSVC
from numpy import array, vstack, ones
from Blob import Blob
import logging

import IAlgorithm


__author__ = 'simon'


class Classificator(IAlgorithm.IAlgorithm):
    def __init__(self, classificator):
        self.model = classificator

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = self.model.predict(blob.data.reshape(1,-1))
            yield blob

    def _train(self, blob_generator):
        # First, collect all elements of the input
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            data.append(blob.data.ravel())
            labels.append(blob.meta.label)
            metas.append(blob.meta)
        try:
            data = vstack(data)
        except ValueError:
            logging.error("Size of all input data needs to be the same for SVM training.")
            raise Exception

        self.model.fit(data, labels)

        for (d,m) in zip(self.model.predict(data),metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b