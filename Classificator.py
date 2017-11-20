from sklearn.svm import LinearSVC
import numpy as np
import scipy
from Blob import Blob
import logging
import time

import IAlgorithm

__author__ = 'simon'


class Classificator(IAlgorithm.IAlgorithm):
    def __init__(self, classificator, use_sparse = None):
        ''' Trains a classificator in training phase and predicts labels in test phase 
        Params:
        @classificator The sklearn classificator having a fit and predict function.
        @use_sparse Whether to store the individual features as sparse matrix.
        If None, sparsity will be used if less space is consumed in sparse format. 
        '''
        self.model = classificator
        logging.debug('Using sparse: %s'%use_sparse)
        self.use_sparse = use_sparse

    def _compute(self, blob_generator):
        for blob in blob_generator:
            blob.data = self.model.decision_function(blob.data.reshape(1,-1))
            yield blob
            
    def _train(self, blob_generator):
        # First, collect all elements of the input
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            if self.use_sparse is None:
                # Determine automatically by comparing size 
                sparse_vec = scipy.sparse.csr_matrix(blob.data.ravel())
                sparse_memory_req = sparse_vec.data.nbytes + sparse_vec.indptr.nbytes + sparse_vec.indices.nbytes
                self.use_sparse = sparse_memory_req < blob.data.nbytes
                logging.debug('Using sparse format for collecting features: %s'%self.use_sparse)
                logging.debug('Blob data needs %i'%blob.data.nbytes)
                logging.debug('%i with sparse vs %i with dense'%(sparse_memory_req,blob.data.nbytes))
            
            if self.use_sparse:
                data.append(scipy.sparse.csr_matrix(blob.data.ravel()))
            else:
                data.append(blob.data.ravel())
            labels.append(blob.meta.label)
            metas.append(blob.meta)
            
        # Stack data to matrix explicitly here, as both fit and predict
        # would to this stacking otherwise
        try:
            if self.use_sparse:
                data = scipy.sparse.vstack(data)
                data = data.astype(np.float64)
            else:
                data = np.array(data, dtype=np.float64)
        except ValueError:
            logging.error("Length of all feature vectors need to be the same for Classificator training.")
            raise Exception
        
        logging.warning('Training the model with feature dim %i, this might take a while'%data.shape[1])
        self.model.fit(data, labels)
        logging.warning('Finished')
    
        for (d,m) in zip(self.model.decision_function(data),metas):
            b = Blob()
            b.data = d
            b.meta = m
            yield b
