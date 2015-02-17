
import unittest
from Dataset import Dataset

class DatasetTest(unittest.TestCase):
    def __init__(self):
        pass

    def test_ICAO(self):
        d = Dataset()
        d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
        d.create_labels_from_path()
        self.assertEqual(d.labels.__len__(),18)
        self.assertEqual(d.classnames_to_labels.__len__(),3)
        self.assertTrue("class1" in d.classnames_to_labels)

    def test_places205(self):
        d = Dataset()
        d.read_from_file("/home/simon/Datasets/places205/val_natural.txt","imagepaths","string")
        d.read_from_file("/home/simon/Datasets/places205/val_natural.txt","labels","int",column=1)
        d.fill_split_assignments(0)


        d.read_from_file("/home/simon/Datasets/places205/train_natural.txt","imagepaths","string")
        d.read_from_file("/home/simon/Datasets/places205/train_natural.txt","labels","int",column=1)
        d.fill_split_assignments(1)
