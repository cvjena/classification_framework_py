import logging
from Classification import Classification
from MeanCalculator import MeanCalculator
from MulticlassSVM import MulticlassSVM

from Dataset import *


__author__ = 'simon'


def run():
    logging.basicConfig(level=logging.DEBUG)
    d = Dataset()
    d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
    d.create_labels_from_path()
    c = Classification(d)
    c.add_algorithm(MeanCalculator())
    c.add_algorithm(MulticlassSVM())
    c.train()
    for path,gt_label in zip(d.imagepaths,d.labels):
        print("Predicted class for "+path+" is "+str(c.predict(path)[0])+" (GT: "+str(gt_label)+")")

if __name__ == "__main__":
    run()