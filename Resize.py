import logging
from scipy.misc import imresize
from skimage.transform import resize
from IAlgorithm import IAlgorithm
import numpy as np

__author__ = 'simon'


class Resize(IAlgorithm):
    def __init__(self, target_shape, mode='stretch'):
        self.target_shape = np.array(target_shape).astype(np.float64)
        self.mode = mode
        
        if mode == 'stretch':
            assert self.target_shape.size >= 2, 'Mode "stretch" requires the target shape to be at least 2 dimensional.'
            
        if mode in ['resize_larger_side','resize_smaller_side']:
            assert isinstance(target_shape,int) or len(target_shape)==1, 'Please provide only a single shape value to Resize if using the mode "resize_{smaller,larger}_side'
            if isinstance(target_shape,int):
                self.target_shape = (self.target_shape, )
                
    def _compute(self, blob_generator):
        for blob in blob_generator:
            if self.mode == 'stretch':
                blob.data = resize(blob.data,self.target_shape,1)
            elif self.mode in ['resize_smaller_side', 'resize_larger_side']:
                # Determine the smaller scaling ration
                if self.mode == 'resize_smaller_side':
                    ratio = self.target_shape / np.min(blob.data.shape[:2])
                if self.mode == 'resize_larger_side':
                    ratio = self.target_shape / np.max(blob.data.shape[:2])
                new_shape = np.round(blob.data.shape[:2] * ratio)
                blob.data = resize(blob.data, new_shape, 1)
            yield blob