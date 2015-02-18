import csv
import logging

from numpy import *
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
            logging.exception("Invalid target field.")

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
                raise Exception("Unsupported datatype")
            items = items[skip_rows:]
            self.__dict__[target_field].extend(items)

    def use_images_in_folder(self, folderpath, filenamepattern=".*\.jpg"):
        for root, subFolders, files in os.walk(folderpath):
            for file in files:
                if re.match(filenamepattern, file):
                    self.imagepaths.append(root + "/" + file)

    # extract class name from directory in which the file is in
    def create_labels_from_path(self):
        if len(self.imagepaths) < 1:
            logging.exception("You need to set the imagepaths before creating the labels")

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
            logging.exception("Size of imagepaths, labels and split assignments are not equal!")
        for path, label, split_assignment in zip(self.imagepaths, self.labels, self.split_assignments):
            b = Blob()
            b.meta.label = label
            b.meta.imagepath = path
            b.meta.split_assignment = split_assignment
            yield b