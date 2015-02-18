import logging

from Classification import Classification
from HOG import HOG
from ImageReader import ImageReader
from MeanCalculator import MeanCalculator
from MulticlassSVM import MulticlassSVM
from Dataset import *
from Resize import Resize
import cProfile
import re

__author__ = 'simon'


def run():
    logging.basicConfig(level=logging.DEBUG)
    d = Dataset()
    d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
    d.create_labels_from_path()
    d.fill_split_assignments(1)
    c = Classification()
    c.add_algorithm(Resize(32,16))
    c.add_algorithm(HOG())
    c.add_algorithm(MulticlassSVM())
    c.train(d)
    for path, gt_label in zip(d.imagepaths, d.labels):
        print("Predicted class for " + path + " is " + str(c.predict(path).data[0]) + " (GT: " + str(gt_label) + ")")


if __name__ == "__main__":
    cProfile.runctx('run()', globals(), locals(), filename="framework.profile")