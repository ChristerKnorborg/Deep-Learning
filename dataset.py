import json
import os
import shutil

from typing import Any
import requests
import zipfile
from pycocotools.coco import COCO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Dataset


from enum import Enum

import torch
from PIL import Image

TRAIN = "./data/train2017"
VALIDATION = "./data/val2017"



# Define the DataSetType enum
class DataSetType(Enum):
    TRAIN = "./data/train2017", "./data/labels/annotations/instances_train2017.json"
    VALIDATION = "./data/val2017", "./data/labels/annotations/instances_val2017.json"
    TEST = "./data/test2017", "./data/labels/annotations/instances_test2017.json"


class DataSetCoco(Dataset):

    def __init__(self, datatype: DataSetType, transform = None):
        """Initializes the dataset. Downloads and extracts data if needed.

        Args:
        - img_dir (str): Path to the directory containing images.
        - ann_file (str): Path to the annotation file.
        """
        self.img_dir, self.ann_file = datatype.value
        self._ensure_data_exists_or_download() # Ensure data exists or download it
        self.coco = COCO(self.ann_file) 
        self.move_images_with_persons_to_person_dir # Move images with persons to the 'person' directory (if not already done)


        self.transform = transform
        self.ids = list(self.coco.imgs.keys())
        self.img_dir = os.path.join(self.img_dir, 'person')




    def _ensure_data_exists_or_download(self):
        """Downloads and extracts the COCO dataset into the data directory if it doesn't already exist."""

        # URLs for TRAIN dataset type
        if self.img_dir == DataSetType.TRAIN.value[0]:
            image_url = "http://images.cocodataset.org/zips/train2017.zip"
            annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # URLs for VALIDATION dataset type
        elif self.img_dir == DataSetType.VALIDATION.value[0]:
            image_url = "http://images.cocodataset.org/zips/val2017.zip"
            annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # URLs for TEST dataset type
        elif self.img_dir == DataSetType.TEST.value[0]:
            image_url = "http://images.cocodataset.org/zips/test2017.zip"
            # No annotations for test set
            annotation_url = None

        else:
            raise ValueError("Unknown data directory.")

        # Construct the zip paths
        image_zip_path = self.img_dir + ".zip"
        annotation_zip_path = "./data/annotations.zip"

        self._download_and_extract(image_url, image_zip_path, self.img_dir)
        self._download_and_extract(
            annotation_url, annotation_zip_path, "./data/labels")

    def _download_and_extract(self, url, save_path, extract_path):
        """Downloads and extracts a zip file from the given URL."""

        if not os.path.exists(extract_path):
            print(f"Downloading from {url} ...")
            response = requests.get(url, stream=True)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Extracting {save_path} ...")
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(save_path)
        else:
            print(f"Files in {extract_path} already exist. Skipping download.")





    def get_categories(self):
        """
        Returns:
        - List of category names.
        """
        categories = self.coco.loadCats(self.coco.getCatIds())
        return [category['name'] for category in categories]

    def get_images_and_annotations(self, categories=[]):
        """
        Returns image paths and annotations for given categories.

        Args:
        - categories (list): List of category names. If empty, fetches data for all categories.

        Returns:
        - List of (image_path, annotations) tuples.
        """
        if not categories:
            category_ids = self.coco.getCatIds()
        else:
            category_ids = self.coco.getCatIds(catNms=categories)

        img_ids = self.coco.getImgIds(catIds=category_ids)

        data = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            print(self.img_dir)
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            ann_ids = self.coco.getAnnIds(
                imgIds=img_id, catIds=category_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            data.append((img_path, anns))

        return data

    def __get_image__(self, __name: str) -> Any:
        pass

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.coco.getImgIds())
    







    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, yolo_targets). 

            yolo_targets is a list of bounding boxes in the format [x, y, w, h, c], where x, y ∈ [0,1] are the horizontal and
            vertical coordinates of the objects midpoint relative to the origin of the bounding box (0,0). 
            E.g. x = 1 means mid point is all the way to the right of the cell.
            The width and height w, h of the object can exceed 1 if the object extends beyond the boundaries of the grid cell.
            c is the confidence score of the bounding box.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.img_dir, path)

        # print("Trying to open:", image_path)
        img = Image.open(image_path).convert('RGB')

        img_width, img_height = img.size  # Get the width and height before transforming (Maybe fix later)

        if self.transform is not None:
            img = self.transform(img)


        # Get the category ID for "person"
        person_cat_id = coco.getCatIds(catNms=["person"])[0]

        # Convert COCO bounding boxes to YOLO format
        yolo_targets = []
        for ann in annotations:
            # Check if the annotation's category ID matches the one for "person"
            if ann['category_id'] == person_cat_id:
                bbox = ann['bbox']
                # Convert top-left (x, y) to center (x_center, y_center)
                x = bbox[0] + bbox[2] / 2
                y = bbox[1] + bbox[3] / 2
                # Convert absolute width and height to relative
                w = bbox[2] / img_width
                h = bbox[3] / img_height
                yolo_targets.append([x, y, w, h])

        return img, yolo_targets
    






    def show_image_with_bboxes(self, index):
        """Displays a random image from the dataset with its bounding boxes."""

        # Choose a image and get its ID and filename
        img_id = self.ids[index]
        img = self.coco.loadImgs([img_id])[0]

        I = plt.imread(os.path.join(self.img_dir, img["file_name"]))  # "file_name is property from object"

        # Load and display instance annotations
        plt.imshow(I)
        annIds = self.coco.getAnnIds(imgIds=img["id"])
        anns = self.coco.loadAnns(annIds)
        print("-----")
        print(anns)
        print("-----")

        # Get the category ID for "person"
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]


        for ann in anns:
            # Check if the annotation's category ID matches the one for "person"
            if ann['category_id'] == person_cat_id:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)

        plt.axis('off')
        plt.show()


    def move_images_with_persons_to_person_dir(self):
        img_dir = self.img_dir  # Image directory
        ann_file = self.ann_file  # Annotation JSON file

        person_img_dir = os.path.join(img_dir, 'person')  # Person image directory

        # Return if the person image directory already exists. Else create it and move images to it
        if os.path.exists(person_img_dir):
            return 
        else:
            os.makedirs(person_img_dir)

        # Read the COCO annotation JSON file for the given dataset type
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Extract category IDs for the 'person' class
        person_cat_id = [cat['id']
                        for cat in data['categories'] if cat['name'] == 'person'][0]

        # Get image ids of images containing persons
        img_ids_with_persons = [ann['image_id']
                                for ann in data['annotations'] if ann['category_id'] == person_cat_id]

        # Deduplicate the image IDs
        img_ids_with_persons = list(set(img_ids_with_persons))

        for i, img_id in enumerate(img_ids_with_persons):
            # Assuming the filenames are formatted as '000000123456.jpg'
            image_name = f"{img_id:012}.jpg"
            image_path = os.path.join(img_dir, image_name)

            new_img_file_path = os.path.join(person_img_dir, image_name)

            print(f"Processing: {i + 1}/{len(img_ids_with_persons)}", end="\r")
            os.rename(image_path, new_img_file_path)

        print(f"Done moving images with persons for dataset!")









# Example of usage:
#coco_data = DataSetCoco(DataSetType.TRAIN)

#coco_data.show_random_image_with_bboxes()
# print(coco_data.get_categories())
# categories = ['person']
# data = coco_data.get_images_and_annotations(categories)
# i = 0
# for img_path, anns in data:
#     if (i >= 1):
#         break

#     print("-------------")
#     print(img_path)
#     print("-------------")
#     print(anns)
#     print("-------------")
#     print(data[i])
#     print("-------------")
#     i += 1

# for index in range(0, 1):
#     print("-------------")
#     print(data[index])
#     print("-------------")
#     print(vars(data[index]))






# TIL AT VISE LABELS FORMAT

# Create an instance of the DataSetCoco class for the TRAIN dataset
coco_data = DataSetCoco(DataSetType.TRAIN)

# Fetch a sample by its index
index_to_test = 5  # You can change this to any valid index
img, yolo_targets = coco_data[index_to_test]

# Print the results
print("Image:", img)
# print image name:
print("Image name:", coco_data.coco.loadImgs(coco_data.ids[index_to_test])[0]['file_name'])
print("Bounding Boxes in YOLO format:", yolo_targets)

get_item = coco_data.__getitem__(index_to_test)
print("Get item:", get_item)
coco_data.show_image_with_bboxes(index_to_test)


