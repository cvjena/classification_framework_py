import csv
import logging

from numpy import *
from numpy.core.umath import logical_and
from numpy.random.mtrand import permutation
import re
import os

from Blob import Blob


__author__ = 'simon'


class Dataset(object):
    def __init__(self):
        self.labels = []
        self.imagepaths = []
        self.classnames_to_labels = {}
        self.split_assignments = []

    def read_from_file(self, filepath, target_field, datatype, column=0, delimiter=" ", skip_rows=0, prepend_string=""):
        if target_field not in self.__dict__:
            logging.error("Invalid target field.")
            raise Exception

        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            items = list(reader)
            items = [it[column] for it in items]
            if target_field == "imagepaths":
                items = [prepend_string + row for row in items]
            elif datatype == "int":
                items = [int32(it) for it in items]
            elif datatype == "float":
                items = [float(it) for it in items]
            else:
                logging.error("Unsupported datatype.")
                raise Exception
            items = items[skip_rows:]
            self.__dict__[target_field].extend(items)

    def use_images_in_folder(self, folderpath, filenamepattern=".*\.(jpg|JPEG|jpeg|png)"):
        for root, subFolders, files in os.walk(folderpath):
            for file in files:
                if re.match(filenamepattern, file):
                    self.imagepaths.append(root + "/" + file)

    # extract class name from directory in which the file is in
    def create_labels_from_path(self):
        if len(self.imagepaths) < 1:
            logging.error("You need to set the imagepaths before creating the labels")
            raise Exception

        self.labels = []
        for path in self.imagepaths:
            class_search = re.search('^(.*[/\\\\])?([^/\\\\]+)[/\\\\][^/\\\\]+$', path)
            if class_search:
                if class_search.group(2) not in self.classnames_to_labels:
                    self.classnames_to_labels[class_search.group(2)] = len(self.classnames_to_labels)
                self.labels.append(self.classnames_to_labels[class_search.group(2)])

    def fill_split_assignments(self, value):
        self.split_assignments.extend(
            [value for _ in range(max(0, len(self.imagepaths) - len(self.split_assignments)))])


    def blob_generator(self):
        if len(self.imagepaths) != len(self.labels) or len(self.labels) != len(self.split_assignments):
            logging.error("Size of imagepaths, labels and split assignments are not equal!")
            raise Exception
        for path, label, split_assignment, idx in zip(self.imagepaths, self.labels, self.split_assignments, range(len(self.labels))):
            if (idx%10==0):
                logging.info("Done with " + str(idx) + " images.")
            b = Blob()
            b.meta.label = label
            b.meta.imagepath = path
            b.meta.split_assignment = split_assignment
            yield b

    def reset_split(self, value):
        self.split_assignments = value * ones((len(self.labels)))


    def make_random_split(self, source_value, target_value, absolute_per_class=1, relative_per_class=0):
        # if num_test is None and num_train is None and percentage_train is None:
        #     logging.error("Your need to provide at lease on of the parameters num_train, num_test or percentage_train!")
        #     raise Exception
        # if num_train is not None and num_test is not None:
        #     logging.error("You can only specify either num_train or num_test!")
        #     raise Exception
        if source_value not in self.split_assignments:
            logging.warning("No element with value 'source_value' in self.split_assignments!")
        if absolute_per_class < 1:
            logging.error("Invalid value for parameter absolute_per_class.")
            raise Exception
        if relative_per_class > 1 or relative_per_class < 0:
            logging.error("Invalid value for parameter relative_per_class.")
            raise Exception

        # for all classes
        classes = unique(self.labels)
        for c in classes:
            class_elements = where(logical_and(transpose(array(self.labels)[newaxis])==c, transpose(array(self.split_assignments)[newaxis])==source_value))[0]
            if len(class_elements)<=absolute_per_class:
                self.split_assignments[class_elements]=target_value
            else:
                how_many = max(absolute_per_class,relative_per_class*len(class_elements))
                self.split_assignments[permutation(class_elements)[:how_many]]=target_value
