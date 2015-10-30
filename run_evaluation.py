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
from SpatialPyramid import SpatialPyramid
from Noise import Noise
import cProfile
import re
import sys
import time
#from Caffe import Caffe
from PyQt4 import QtGui, QtCore

__author__ = 'simon'


def run():
    logging.getLogger().setLevel(logging.WARNING)
    d = Dataset()
    #d.use_images_in_folder("/home/simon/Datasets/ImageNet_Natural/images/")
    #d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
    d.use_images_in_folder("/home/simon/Datasets/desko_ids/images_unique/")
    #d.use_images_in_folder("/home/simon/Datasets/croatianFishDataset-final/")
    #d.use_images_in_folder("/home/jaeger/data/croatianFishDataset1-5Dir/")
    d.create_labels_from_path()
    d.fill_split_assignments(1)

    #d.read_from_file("/home/simon/Datasets/CUB_200_2011/cropped_scaled_alex.txt","imagepaths","string")
    #d.read_from_file("/home/simon/Datasets/CUB_200_2011/tr_ID.txt","split_assignments","int")
    #d.read_from_file("/home/simon/Datasets/CUB_200_2011/labels.txt","labels","int")

    c = Classification()
    c.add_algorithm(ImageReader())
    c.add_algorithm(Resize(10,6))
    # #c.add_algorithm(Noise('saltpepper',0.1))
    #p = ParallelAlgorithm()
    #
    # p1 = AlgorithmPipeline()
    # #p1.add_algorithm(Resize(128,64))
    #c.add_algorithm(HOG())
    #c.add_algorithm(SpatialPyramid())
    # #p1.add_algorithm(MinMaxNormalize())
    # c.add_algorithm(NormNormalize())
    # p.add_pipeline(p1)
    #
    #p2 = AlgorithmPipeline()
    #p2.add_algorithm(Resize(64,32))
    #p2.add_algorithm(Colorname())
    #p2.add_algorithm(SpatialPyramid())
    #p2.add_algorithm(NormNormalize())
    # #p2.add_algorithm(MinMaxNormalize())
    #p.add_pipeline(p2)
    #
    #c.add_algorithm(p)
    # #c.add_algorithm(MinMaxNormalize())
    # #c.add_algorithm(NormNormalize())
    # c.add_algorithm(MeanCalculator())
    c.add_algorithm(MulticlassSVM())
    # #c.train(d)
    # #for path, gt_label in zip(d.imagepaths, d.labels):
    # #    logging.info("Predicted class for " + path + " is " + str(c.predict(path).data[0]) + " (GT: " + str(gt_label) + ")")

    ## Caffe features
    #c.add_algorithm(Caffe("","","fc7"))
    #c.add_algorithm(MulticlassSVM())

    #with open('run_evaluation.py', 'r') as fin:
    #    print(fin.read())

    mean_acc,mean_mAP = Evaluation.random_split_eval(d,c,absolute_train_per_class=2,runs=1)
    #mean_acc,mean_mAP = Evaluation.fixed_split_eval(d,c)
    logging.warning("Total accuracy is " + str(mean_acc))
    logging.warning("Total mAP is " + str(mean_mAP))


if __name__ == "__main__":
    t = time.time()
    cProfile.runctx('run()', globals(), locals(), filename="framework.profile")
    print("Done in " + str(time.time()-t))
    #run()