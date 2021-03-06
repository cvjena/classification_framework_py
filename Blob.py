from numpy import array
import uuid

__author__ = 'simon'


class BlobMeta(object):
    def __init__(self):
        self.imagepath = ""
        self.label = array([])
        self.split_assignment = -1
        self.uuid = str(uuid.uuid4())


class Blob(object):
    def __init__(self):
        self.data = array([])
        self.meta = BlobMeta()