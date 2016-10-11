import logging
import numpy
from numpy import nanmean, mean
import numpy as np
from sklearn.metrics import confusion_matrix
import pyprind

__author__ = 'simon'

class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def fixed_split_eval(dataset, classification):
        return Evaluation._eval(dataset,classification,1,0)

    @staticmethod
    def random_split_eval(dataset, classification, runs=1, absolute_train_per_class=None, relative_train_per_class=None,absolute_test_per_class=None):
        if 1 != numpy.sum([absolute_train_per_class is not None, relative_train_per_class is not None, absolute_test_per_class is not None]):
            logging.error("You need to pass exactly one of the parameters absolute_train_per_class, relative_train_per_class,absolute_test_per_class")
            raise Exception

        accs = []
        mAPs = []

        for run_id in range(runs):
            if absolute_train_per_class is not None:
                # First, mark all as test
                dataset.reset_split(0)
                # Now select train images
                dataset.make_random_split(0,1,absolute_train_per_class,0)
            elif relative_train_per_class is not None:
                # First, mark all as test
                dataset.reset_split(0)
                # Now select train images
                dataset.make_random_split(0,1,1,relative_train_per_class)
            elif absolute_test_per_class:
                # First, mark all as train
                dataset.reset_split(1)
                # Now select test images
                dataset.make_random_split(1,0,absolute_test_per_class,relative_train_per_class)
            (acc,mAP) = Evaluation._eval(dataset, classification, 1, 0)
            accs.append(acc)
            mAPs.append(mAP)

        mean_acc = mean(accs)
        mean_mAP = mean(mAPs)
        return (mean_acc,mean_mAP)

    @staticmethod
    def _eval(dataset, classification, train_split, test_split):
        train_blobs = [b for b in dataset.blob_generator() if b.meta.split_assignment==train_split]
        test_blobs = [b for b in dataset.blob_generator() if b.meta.split_assignment==test_split]
        
        # Train pipeline
        train_output = list(classification.pipeline.train(pyprind.prog_bar(train_blobs)))
        train_labels = [blob.meta.label for blob in train_output]
        id_label_mapping = np.unique(train_labels)
        # Test pipeline
        test_output = list(classification.pipeline.compute(pyprind.prog_bar(test_blobs)))
        test_labels = [b.meta.label for b in test_output]
        test_pred = [id_label_mapping[blob.data[0].argmax()] for blob in test_output]
        
        # Compute confusion matrix
        
        cm = confusion_matrix(test_labels, test_pred)
        acc = cm.diagonal().sum()/cm.ravel().sum()
        cm = cm / cm.sum(axis=1,keepdims=True)
        mAP = nanmean(cm.diagonal())
        logging.warning("Accuracy is " + str(acc))
        logging.warning("ARR is " + str(mAP))
        return (acc,mAP)


