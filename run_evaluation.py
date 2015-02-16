import Dataset
import Classification
import logging

__author__ = 'simon'


logging.basicConfig(level=logging.DEBUG)
d = Dataset()
d.use_images_in_folder("/home/simon/Datasets/ICAO_german/")

c = Classification()
