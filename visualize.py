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
    index = int(sys.argv[1])

    colors = {
        1: (0,255,0), # Swimmer 
        2: (255,0,0), # Floater
        3: (51,255,255), # Boat
        4: (255,255,0), # Swimmer on Boat
        5: (255, 102, 255), # Floater on Boat
        6: (255, 153, 51)} # Life Jacket
    
    # If input is within the valid range, proceed
    if (index >= 0) and (index <= len(mapping)):
        
        # Load image as an np.array
        image = np.array(Image.open(os.path.join(data_path, f_names[mapping[index]])))
        
        print(annotations_swimmers["images"][:10])
        # Iterate over all annotations
        for an in annotations_swimmers['annotations']:
            # If image_id same as the image we are looking for, fetch bbox
            if an['image_id'] == mapping[index - 1]:
                print(an)
                print(annotations_swimmers["images"][an["image_id"]]["file_name"])
                print(mapping[0:3])
                bbox = an['bbox']
                category_id = an['category_id']
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = x1 + bbox[2]
                y2 = y1 + bbox[3]
                # Draw bounding box on image
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[category_id], 2)
        
        # Display the image after all the bounding boxes have been stored
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        
if __name__ == '__main__':
    visualize()