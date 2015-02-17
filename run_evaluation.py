from Dataset import *
from Classification import *
import logging
from tests.DatasetTest import DatasetTest

__author__ = 'simon'

def run():
    logging.basicConfig(level=logging.DEBUG)
    d = Dataset()
    d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")
    d.create_labels_from_path()

    d = Dataset()
    d.read_from_file("/home/simon/Datasets/places205/val_natural.txt","imagepaths","string")
    d.read_from_file("/home/simon/Datasets/places205/val_natural.txt","labels","int",column=1)
    d.fill_split_assignments(0)


    d.read_from_file("/home/simon/Datasets/places205/train_natural.txt","imagepaths","string")
    d.read_from_file("/home/simon/Datasets/places205/train_natural.txt","labels","int",column=1)
    d.fill_split_assignments(1)
    pass

if __name__ == "__main__":
    run()