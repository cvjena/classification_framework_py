from sklearn.svm import LinearSVC
import numpy as np
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
            
        # Stack data to matrix explicitly here, as both fit and predict
        # would to this stacking otherwise
        try:
            data = np.vstack(data)
        except ValueError:
            logging.error("Length of all feature vectors need to be the same for Classificator training.")
            raise Exception
        
        logging.warning('Training the model, this might take a while')
        self.model.fit(data, labels)
    
        for (d,m) in zip(self.model.predict(data),metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b