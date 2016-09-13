from IAlgorithm import IAlgorithm
import numpy as np

__author__ = 'simon'


class Pad(IAlgorithm):
    ''' Pads the input array using numpy.pad '''
    def __init__(self, target_width = None, target_height = None):
        self.target_width = target_width
        self.target_height = target_height
        
    def _compute(self, blob_generator):
        for blob in blob_generator:
            if self.target_height:
                offset_height = int((self.target_height - blob.data.shape[0]) / 2)
                assert offset_height >= 0, 'Input image larger than target height'
                target_height = self.target_height
            else:
                offset_height = 0
                target_height = blob.data.shape[0]
            if self.target_width:
                offset_width = int((self.target_width - blob.data.shape[1]) / 2)
                assert offset_width >= 0, 'Input image larger than target width'
                target_width = self.target_width
            else:
                offset_width = 0
                target_width = blob.data.shape[1]
                
            embed = np.zeros((target_height,target_width) + blob.data.shape[2:],blob.data.dtype)
            embed[offset_height : offset_height + blob.data.shape[0],
                  offset_width : offset_width + blob.data.shape[1],...] = blob.data
            blob.data = embed
            yield blob