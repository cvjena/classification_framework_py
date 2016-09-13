from IAlgorithm import IAlgorithm
import numpy as np

__author__ = 'simon'


class Crop(IAlgorithm):
    ''' Crop '''
    def __init__(self, target_shape, mode = 'center'):
        assert mode in ['center', 'random'], "Mode %s is not know for layer 'crop'"%str(mode)
        self.target_shape = np.array(target_shape)
        self.mode = mode
        
    def _compute(self, blob_generator):
        for blob in blob_generator:
            assert len(self.target_shape) <= len(blob.data.shape)
            if self.mode == 'random':
                offset = (np.random.rand(*self.target_shape.shape) * (np.array(blob.data.shape[:len(self.target_shape)]) - self.target_shape)).astype(int)
            elif self.mode == 'center':
                offset = ((np.array(blob.data.shape[:len(self.target_shape)]) - self.target_shape)/2).astype(int)
            assert np.all(offset>=0), "Input array to Crop is too small with shape %s and target shape %s"%(str(blob.data.shape),str(self.target_shape))
            indices = [np.s_[off:off+sz] for off,sz in zip(offset,self.target_shape)]
            indices += [np.s_[:] for _ in range(len(blob.data.shape) - len(self.target_shape))]
            blob.data = blob.data[indices]
            yield blob