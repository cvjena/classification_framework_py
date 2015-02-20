from sklearn.svm import LinearSVC
from numpy import array, vstack, ones
from Blob import Blob
import logging

import IAlgorithm


__author__ = 'simon'


class MulticlassSVM(IAlgorithm.IAlgorithm):
    def __init__(self, penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0, random_state=None):
        self.svm_model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                                   fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                   class_weight=class_weight, verbose=verbose, random_state=random_state)

    def _compute(self, blob_generator):
        for blob in blob_generator:
            flat_data=blob.data.ravel()
            # Add 1 as feature for bias in SVM
            d = ones((1,flat_data.shape[0]+1))
            d[:,:-1] = flat_data
            blob.data = self.svm_model.predict(self._add_bias(blob.data.ravel()))
            yield blob

    def _train(self, blob_generator):
        # First, collect all elements of the input
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            data.append(self._add_bias(blob.data.ravel()))
            labels.append(blob.meta.label)
            metas.append(blob.meta)
        try:
            data = vstack(data)
        except ValueError:
            logging.error("Size of all input data needs to be the same for SVM training.")
            raise Exception

        self.svm_model.fit(data, labels)

        for (d,m) in zip(self.svm_model.predict(data),metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b

    def _add_bias(self, flattened_vector):
        d = ones((1,flattened_vector.shape[0]+1))
        d[:,:-1] = flattened_vector
        return d