from sklearn.svm import LinearSVC
import numpy as np
import scipy
from Blob import Blob
import logging

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
            blob.data = self.model.predict(blob.data.reshape(1,-1))
            yield blob
            
    def _train(self, blob_generator):
        # First, collect all elements of the input
        data = []
        labels = []
        metas = []
        for blob in blob_generator:
            if self.use_sparse is None:
                # Determine automatically by comparing size 
                sparse_vec = scipy.sparse.csr_matrix(blob.data.ravel().astype(np.float64))
                # We use sparse, if sparse data 
                # In case of sparse, we need to store this sparse matrix only, no conversation need
                sparse_memory_req = sparse_vec.data.nbytes + sparse_vec.indptr.nbytes + sparse_vec.indices.nbytes
                # In case of dense, we need to collect all blobs in a list first and the a dense float64 matrix
                # will be generated -> 1 blob.data list + 1 dense float64 matrix will be stored at peak
                dense_memory_req = blob.data.nbytes + 8*blob.data.size
                self.use_sparse = sparse_memory_req < dense_memory_req
                logging.debug('Using sparse format for collecting features: %s'%self.use_sparse)
                logging.debug('Blob data needs %i'%blob.data.nbytes)
                logging.debug('%i with sparse vs %i with dense'%(sparse_memory_req,dense_memory_req))
            
            if self.use_sparse:
                if isinstance(data, list):
                    data = scipy.sparse.csr_matrix(blob.data.ravel().astype(np.float64))
                else:
                    # Append row to csr matrix
                    # This is a bit hacky but prevents copying
                    new_row = scipy.sparse.csr_matrix(blob.data.ravel().astype(np.float64))
                    data.data = np.hstack((data.data,new_row.data))
                    data.indices = np.hstack((data.indices,new_row.indices))
                    data.indptr = np.hstack((data.indptr,(new_row.indptr + data.nnz)[1:]))
                    data._shape = (data.shape[0]+new_row.shape[0],new_row.shape[1])
            else:
                data.append(blob.data.ravel())
            labels.append(blob.meta.label)
            metas.append(blob.meta)
            
        if not self.use_sparse:
            # Stack data to matrix explicitly here, as both fit and predict
            # would to this stacking otherwise
            try:
                data = np.array(data, dtype=np.float64)
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