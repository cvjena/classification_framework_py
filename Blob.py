from numpy import array
import uuid

__author__ = 'simon'


class BlobMeta(object):
    def __init__(self):
        self.imagepath = ""
        self.label = array([])
        self.split_assignment = -1
        self.uuid = uuid.uuid4()
    def __reduce__(self):
        return (BlobMeta, ())


class Blob(object):
    def __init__(self):
        self.data = array([])
        self.meta = BlobMeta()
    def __reduce__(self):
        return (Blob, ())