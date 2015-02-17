import logging

__author__ = 'simon'


class IAlgorithm(object):
    def __init__(self):
        pass

    def compute(self, in_array):
        logging.info("Using " + str(type(self)) + " compute function.")
        raise NotImplementedError("Not implemented.")

    def compute_all(self, data_generator):
        logging.info("Using " + str(type(self)) + " compute_all function.")
        for in_array in data_generator:
            yield self.compute(in_array)

    def train(self, data_generator, label_generator):
        logging.info("Using " + str(type(self)) + " training function.")
        return self.compute_all(data_generator)

    def requires_training(self):
        logging.info("Training required: " + str(
            getattr(self, "requires_training").__func__ == getattr(IAlgorithm(), "requires_training").__func__))
        return getattr(self, "requires_training").__func__ == getattr(IAlgorithm(), "requires_training").__func__