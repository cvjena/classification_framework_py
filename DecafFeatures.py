#import logging
#from numpy.core.umath import arctan2, cos, sin
#from skimage import color
from IAlgorithm import IAlgorithm
from Blob import Blob
import numpy as np
#from scipy import sqrt
#from scipy.constants import pi
#from scipy.ndimage import uniform_filter
from scipy.misc import imresize
from decaf.scripts.jeffnet import JeffNet


__author__ = 'jaeger'


class DecafFeatures(IAlgorithm):
    def __init__(self,model,meta,layer):
        self.model = model
        self.meta = meta
        self.net = JeffNet(self.model, self.meta)
        self.layer = layer

    def _compute(self, blob_generator):
        for blob in blob_generator:
            image = blob.data

            blob.data = self.decafFeatures(image)
            yield blob

    def decafFeatures(self, image):
	img = imresize(img,(256,256,3),interp='bicubic')
        score = self.net.classifiyWholeImage(img)
        if self.layer=="score":
            return list(score)
        else:
            return list(self.net.feature(self.layer)[0].reshape((1,-1))[0])
            
#def __init__(self,model,meta,layer):
#        self.model = model
#        self.meta = meta
#        self.net = JeffNet(self.model, self.meta)
#        self.layer = layer
#    def extractFromFile(self,file_path):
#        # Calculate features
#        img = pylab.imread(file_path)
#        img = imresize(img,(256,256,3),interp='bicubic')
#        score = self.net.classifiyWholeImage(img)
#        if self.layer=="score":
#            return list(score)
#        else:
#            return list(self.net.feature(self.layer)[0].reshape((1,-1))[0])
#    def __str__(self):
#        s="Decaf Feature Extractor\n"
#        s+="Model Data: %s\n"%(self.model)
#        s+="Meta Data: %s\n"%(self.meta)
#        s+="Layer for features: %s\n"%(self.layer)
#        return s