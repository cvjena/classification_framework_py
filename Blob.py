from numpy import array

__author__ = 'simon'


class BlobMeta(object):
    def __init__(self):
        self.imagepath = ""
        self.label = array([])
        self.split_assignment = -1


class Blob(object):
    def __init__(self):
        self.data = array([])
        self.meta = BlobMeta()