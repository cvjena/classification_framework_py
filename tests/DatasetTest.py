import unittest

from numpy import *

from Dataset import Dataset


class DatasetTest(unittest.TestCase):
    def test_ICAO(self):
        d = Dataset()
        d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
        d.create_labels_from_path()
        self.assertEqual(d.labels.__len__(), 18)
        self.assertEqual(d.classnames_to_labels.__len__(), 3)
        self.assertTrue("class1" in d.classnames_to_labels)

    def test_places205(self):
        d = Dataset()
        d.read_from_file("/home/simon/Datasets/places205/val_natural.txt", "imagepaths",
                         "string", prepend_string="/home/simon/Datasets/places205/images/images256/")
        d.read_from_file("/home/simon/Datasets/places205/val_natural.txt", "labels", "int", column=1)
        d.fill_split_assignments(0)

        d.read_from_file("/home/simon/Datasets/places205/train_natural.txt", "imagepaths",
                         "string", prepend_string="/home/simon/Datasets/places205/images/images256/")
        d.read_from_file("/home/simon/Datasets/places205/train_natural.txt", "labels", "int", column=1)
        d.fill_split_assignments(1)

        self.assertEqual(len(d.split_assignments), 201448)
        self.assertEqual(len(d.labels), 201448)
        self.assertEqual(len(d.imagepaths), 201448)
        self.assertTrue((array(d.split_assignments[1:1400]) == 0).all())
        self.assertTrue((array(d.split_assignments[1401:]) == 1).all())
        self.assertTrue((unique(array(d.labels)) == array(range(13))).all())
