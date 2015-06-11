import logging
import numpy
from numpy import nanmean, mean
from sklearn.metrics import confusion_matrix

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
        train_labels = [blob.meta.label for blob in classification.pipline.train((blob for blob in dataset.blob_generator() if blob.meta.split_assignment==train_split))]
        test_labels = [blob.meta.label for blob in dataset.blob_generator() if blob.meta.split_assignment==test_split]
        test_predictions = [blob.data[0] for blob in classification.compute((blob for blob in dataset.blob_generator() if blob.meta.split_assignment==test_split))]

        # Compute confusion matrix
        cm = confusion_matrix(test_labels, test_predictions)
        acc = cm.diagonal().sum()/cm.ravel().sum()
        logging.warning("Accuracy is " + str(acc))
        cm = cm / cm.sum(axis=1,keepdims=True)
        mAP = nanmean(cm.diagonal())
        logging.warning("mAP is " + str(mAP))
        return (acc,mAP)


