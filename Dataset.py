from numpy import *
import csv
import re
import os
import logging

__author__ = 'simon'

class Dataset:
    labels = array([])
    imagepaths = []
    classnames_to_labels = {}
    split_assignments = array([])

    def read_from_file(self, filepath, target_field, datatype, column=0, delimiter=" ", skip_rows=0, basedir="/"):
        if target_field not in self.props():
            raise Exception("Invalid target field.")

        if datatype == "float" or datatype == "int":
            items = loadtxt(open(filepath, "r"), delimiter=delimiter, skiprows=skip_rows)
            items = items[:,column]
            self.props()[target_field] = self.props()[target_field].append(items)
        elif datatype == "string":
            with open('file.csv', 'rb') as f:
                reader = csv.reader(f)
                items = list(reader)
                items = items[skip_rows:]
                items = [basedir + row[column] for row in items]
                self.props()[target_field] = self.props()[target_field] + items
        else:
            raise Exception("Unsupported datatype")

    def use_images_in_folder(self, folderpath, filenamepattern=".*\.jpg"):
        for root, subFolders, files in os.walk(folderpath):
            for file in files:
                if re.match(filenamepattern, file):
                    self.imagepaths.append(root+ "/" + file)
                    logging.info("Added file "+file)

    # extract class name from directory in which the file is in
    def create_labels(self):
        if len(self.imagepaths)<1:
            raise Exception("You need to set the imagepaths before creating the labels")

        self.labels = array([])
        for path in self.imagepaths:
            class_search = re.search('^(.*[/\\\\])?([^/\\\\]+)[/\\\\][^/\\\\]+$', path)
            if class_search:
                if class_search.group(2) not in self.classnames_to_labels:
                    self.classnames_to_labels[class_search.group(2)] = len(self.classnames_to_labels)
                self.labels.append(self.classnames_to_labels[class_search.group(2)])
