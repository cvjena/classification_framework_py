from IAlgorithm import IAlgorithm
import caffe
import numpy as np
import itertools
import sys
import logging

__author__ = 'simon'


class Caffe(IAlgorithm):
    def __init__(self, 
                 proto = '/home/simon/Data/CNN/vgg-m/deploy.prototxt', 
                 weights = '/home/simon/Data/CNN/vgg-m/model', 
                 mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                 inblob = 'data',
                 outblob = 'fc7',
                 endlayer = None,
                 batchsize = 1):
        ''' 
        Loads a cnn model and prepares everything for the forward pass
        @proto Caffe protobuf text message describing the architecture
        @weigths Trained model containing the weights for the layers described in proto
        @mean The dataset dependent image mean, usually it is safe to leave it at the ImageNet mean
        @inblob The name of the layer to write the image to
        @outblob The name of the layer, where we shall take the activations from and return it as result
        @endlayer The layer where we shall stop the forward pass
        ''' 
        
        # Use caffe model for feature extraction
        self.net = caffe.Classifier(proto, weights, mean = mean, 
                                    channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
        self.proto = proto
        self.weights = weights
        self.mean = mean
        assert inblob in self.net.blobs.keys(), 'Blob %s does not exist'%inblob
        self.inblob = self.net.blobs[inblob]
        self.set_output(outblob, endlayer)
        assert batchsize > 0, 'Batch size has to be greater or equal to 0'
        self.inblob.reshape(batchsize, *self.inblob.data.shape[1:])
        
    def preprocess(self, img):
        return (np.rollaxis(img, 2)[::-1]) - self.net.transformer.mean['data']

    def _compute(self, blob_generator):
        finished = False
        while not finished:
            blobs = list(itertools.islice(blob_generator, self.inblob.data.shape[0]))
            if len(blobs) > 0:
                # TODO: maybe the input is not properly shaped?
                for idx, b in enumerate(blobs):
                    if b.data.dtype == np.uint8:
                        data = np.float32(b.data)
                    elif b.data.dtype in [np.float32, np.float64]:
                        data = 255 * b.data
                    else:
                        raise Exception('Unknown type')
                        
                    if data.shape[:2] != self.inblob.data.shape[2:]:
                        logging.warning(('Warning: data shape does not match CNN input size, '
                            'reshaping CNN input to %s, which might cause crash.')
                            %(str((self.inblob.shape[0], data.shape[2], data.shape[0], data.shape[1]))))
                        sys.stdout.flush()
                        self.inblob.reshape(self.inblob.shape[0],
                                            data.shape[2],
                                            data.shape[0],
                                            data.shape[1])
                    self.inblob.data[idx,...] = self.preprocess(data)
                self.net.forward(end=self.endlayer)
                for idx,b in enumerate(blobs):
                    b.data = self.outblob.data[idx,...].copy()
                    yield b
            else:
                finished = True
            
    def get_net(self):
        return self.net
    
    def set_output(self, outblob, endlayer):
        assert outblob in self.net.blobs.keys(), 'Blob %s does not exist'%outblob
        if outblob:
            self.outblob = self.net.blobs[outblob]
        else:
            assert endlayer in self.net.blobs.keys(), 'Neither outblob is defined nor does endlayer exists as a blob'
            self.outblob = self.net.blobs[endlayer]
        assert endlayer is None or endlayer in list(self.net._layer_names), 'Layer %s does not exist'%endlayer
        self.endlayer = endlayer
