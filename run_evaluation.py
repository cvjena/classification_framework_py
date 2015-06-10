import logging
from AlgorithmPipeline import AlgorithmPipeline

from Classification import Classification
from Colorname import Colorname
from Evaluation import Evaluation
from HOG import HOG
from ImageReader import ImageReader
from MeanCalculator import MeanCalculator
from MinMaxNormalize import MinMaxNormalize
from MulticlassSVM import MulticlassSVM
from Dataset import *
from NormNormalize import NormNormalize
from ParallelAlgorithm import ParallelAlgorithm
from Resize import Resize
import cProfile
import re
from SpatialPyramid import SpatialPyramid

__author__ = 'simon'


def run():
    logging.basicConfig(level=logging.INFO)
    d = Dataset()
    #d.use_images_in_folder("/home/simon/Datasets/ImageNet_Natural/images/")
    d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
    #d.use_images_in_folder("/home/simon/Datasets/desko_ids/images_unique/")
    d.create_labels_from_path()
    d.fill_split_assignments(1)
    c = Classification()
    p1 = AlgorithmPipeline()
    p2 = AlgorithmPipeline()
    p1.add_algorithm(Resize(512,320))
    p1.add_algorithm(HOG())
    p1.add_algorithm(SpatialPyramid())
    #p1.add_algorithm(MinMaxNormalize())
    p2.add_algorithm(Resize(64,32))
    p2.add_algorithm(Colorname())
    p2.add_algorithm(SpatialPyramid())
    #p2.add_algorithm(MinMaxNormalize())
    p = ParallelAlgorithm()
    p.add_pipeline(p2)
    p.add_pipeline(p1)
    c.add_algorithm(p)
    c.add_algorithm(MinMaxNormalize())
    c.add_algorithm(MulticlassSVM())
    #c.train(d)
    #for path, gt_label in zip(d.imagepaths, d.labels):
    #    logging.info("Predicted class for " + path + " is " + str(c.predict(path).data[0]) + " (GT: " + str(gt_label) + ")")
    Evaluation.random_split_eval(d,c,absolute_train_per_class=1,runs=10)


if __name__ == "__main__":
    cProfile.runctx('run()', globals(), locals(), filename="framework.profile")