import IAlgorithm

__author__ = 'simon'


class SampleAlgorithm(IAlgorithm.IAlgorithm):
    def __init__(self):
        pass

    def compute(self, in_array):
        return in_array

    # Optional, only makes sense if you want to program a (faster) batch processing variant
    def compute_all(self, in_array_generator):
        # Recommended design pattern:
        # Use yield!
        # for in_array in in_array_generator:
        #     yield self.compute(in_array)
        return IAlgorithm.compute_all(self,in_array_generator)

    # Optional, only required if your algorithm needs training, otherwise remove this function
    def train(self, in_array_generator, label_generator):
        # Use yield if the images are processed one by one!
        return self.compute_all(in_array_generator)