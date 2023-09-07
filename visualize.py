import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

def visualize():
    data_path = "data/train"
    annotation_path_obj = "data/annotations/instances_train_objects_in_water.json"
    annotation_path_swimmers = "data/annotations/instances_train_swimmer.json"

    # Empty lists and dicts to hold training data and labels
    f_names: dict = {}
    
    # Get all the file names in train dir and seperate extensions and discard them
    for file in os.listdir(path=data_path):
        f_names[int(file.split(".")[0])] = file

    # Sort f_names in numerical order to maintain sequencing
    mapping = sorted(list(f_names.keys()))

    # Opening objects in water JSON file and loading data
    with open(annotation_path_obj) as json_file_obj:
        annotations_objects = json.load(json_file_obj)

    # Opening swimmers in water JSON file and loading data
    with open(annotation_path_swimmers) as json_file_swim:
        annotations_swimmers = json.load(json_file_swim)

    # Display a given image with its annotations
    index = int(sys.argv[1]) - 1

if __name__ == '__main__':
    visualize()