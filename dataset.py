import json
import os
from random import random
import shutil
from torchvision import transforms
from typing import Any
import requests
import zipfile
from pycocotools.coco import COCO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from torch.utils.data import Dataset

from enum import Enum
from PIL import Image

import random
TRAIN = "./data/train2017"
VALIDATION = "./data/val2017"


from model_constants import S, B, C # Import grid dimension S = 7, Bounding boxes per cell B = 2, and classes C = 1



# Define the DataSetType enum
class DataSetType(Enum):
    TRAIN = "./data/train2017", "./data/labels/annotations/instances_train2017.json"
    VALIDATION = "./data/val2017", "./data/labels/annotations/instances_val2017.json"
    TEST = "./data/test2017", "./data/labels/annotations/instances_test2017.json"


class DataSetCoco(Dataset):

    def __init__(self, datatype: DataSetType, transform = None, save_crop = False):
        """Initializes the dataset. Downloads and extracts data if needed.

        Args:
        - datatype (DataSetType): The type of dataset to use consisting of:
            - img_dir (str): Path to the directory containing images.
            - ann_file (str): Path to the annotation file.
        - transform (callable, optional): Optional transform to be applied 
        """
        self.img_dir, self.ann_file = datatype.value
        self._ensure_data_exists_or_download() # Ensure data exists or download it
        self.coco = COCO(self.ann_file) 
        self.move_images_with_persons_to_person_dir # Move images with persons to the 'person' directory (if not already done)


        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

        self.save_crop = save_crop # If True, saves the last crop coordinates to self.last_crop_coordinates
        self.last_crop_coordinates = [] # Stores the last crop coordinates if save_crop is True
        








##############################################################################################################
                        ### FILE DOWNLOADING AND ORGANIZATION CODE STARTS HERE ###
##############################################################################################################





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




##############################################################################################################
                                ### PRODUCTION CODE STARTS HERE ###
##############################################################################################################

    



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




    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.coco.getImgIds())



    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.img_dir, path)

        # Check if the image exists in the train2017 folder. If not, change the path to look inside the 'person' subfolder
        if not os.path.exists(image_path):
            # Modify the path to look inside the 'person' subfolder
            image_path = os.path.join(self.img_dir, 'person', path)

        
        img = Image.open(image_path).convert('RGB')

        # Convert the PIL Image to a tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)

        # Crop the image
        img, bounding_boxes = self.crop_image(img, annotations)

        original_img_width, original_img_height = img.shape[1:3]

        if self.transform is not None:
            img= self.transform(img)

        img_height, img_width = img.shape[1:3]  # Get the width and height after transforming the image (now in tensor format)



        # Calculate scaling factors
        width_scale = img_width / original_img_width
        height_scale = img_height / original_img_height



        # The image is divided into a grid of size S x S. with each cell being of size C + 5, where C is the number of classes.
        label_tensor = torch.zeros((S, S, 5 + C))


        # Convert COCO bounding boxes to YOLO format
        for bbox in bounding_boxes:
            # Only create labels for category that matches the "person" class

            # Adjust for resizing
            bbox[0] *= width_scale
            bbox[1] *= height_scale
            bbox[2] *= width_scale
            bbox[3] *= height_scale

            # Find midpoint coordinate (x, y) of bounding box
            x_center = bbox[0] + bbox[2] / 2
            y_center  = bbox[1] + bbox[3] / 2

            # Normalize the coordinates
            x = x_center / img_width
            y = y_center / img_height

            # Convert absolute width and height to be relative to the total image dimensions
            w = bbox[2] / img_width
            h = bbox[3] / img_height


            # Determine grid cell
            i, j = int(y * S), int(x * S)
            x_cell_offset = x*S - j # Offset of midpoint x coordinate from the left side of the cell
            y_cell_offset = y*S - i # Offset of midpoint y coordinate from the top side of the cell



            # Update cell values with the formula (C_1, C_2, ... , C_n, P_c, x, y, w, h), where C_i is the probability of the object being class i (Class score),
            # P_c is the probability of an object being present in the cell (Objectness score), and (x, y) are the coordinates of the midpoint of the bounding box relative to its grid cell, 
            # and (w, h) are the width and height of the bounding box relative to the whole image.

            if label_tensor[i, j, C] == 0: # If there's no object in the cell yet
                
                # Assuming there's only one class ("person") for now. 
                # We would meed to loop over classes and set them accordingly if there were more.
                label_tensor[i, j, 0] = 1 # Class score for "person" (probability of the object being a "person")

                # Set objectness score
                label_tensor[i, j, C] = 1  # Objectness score

                # Set bounding box attributes
                label_tensor[i, j, C + 1:C + 5] = torch.tensor([x_cell_offset, y_cell_offset, w, h])
            
            else: # Choose the bounding box with the largest area
                
                # Get the current bounding box attributes
                current_w = label_tensor[i, j, C + 3]
                current_h = label_tensor[i, j, C + 4]

                # Calculate the area of the current and new bounding box
                current_area = current_w * current_h
                new_area = w * h

                # If the new bounding box has a larger area, replace the old bounding box with the new one
                if new_area > current_area:

                    # Assuming there's only one class ("person") for now. 
                    # We would meed to loop over classes and set them accordingly if there were more.
                    label_tensor[i, j, 0] = 1 # Class score for "person" (probability of the object being a "person")

                    # Set objectness score
                    label_tensor[i, j, C] = 1  # Objectness score

                    # Set bounding box attributes
                    label_tensor[i, j, C + 1:C + 5] = torch.tensor([x_cell_offset, y_cell_offset, w, h])
                

        return img, label_tensor
    



    def crop_image(self, img: torch.Tensor, annotations, size=(256, 256)):
        # Get the category ID for "person"
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]
        bounding_boxes = []

        # Randomly choose a top-left corner for cropping
        width, height = size
        max_x = img.shape[2] - width
        max_y = img.shape[1] - height
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        x2 = x1 + width
        y2 = y1 + height

        # Save crop coordinates if save_crop flag is set and they are not already saved.
        if self.save_crop and len(self.last_crop_coordinates) == 0:
            self.last_crop_coordinates = (x1, y1, x2, y2)
        elif self.save_crop:
            x1, y1, x2, y2 = self.last_crop_coordinates


        # Crop the image
        new_img = img[:, y1:y2, x1:x2]



        # Define the corners for the crop
        crop_top_left = (x1, y1)
        crop_top_right = (x2, y1)
        crop_bottom_left = (x1, y2)
        crop_bottom_right = (x2, y2)

        


        for ann in annotations:
            bbox = ann['bbox']
            if ann['category_id'] == person_cat_id:


                # Define the corners for the bounding box
                bbox_top_left = (bbox[0], bbox[1])
                bbox_top_right = (bbox[0] + bbox[2], bbox[1])
                bbox_bottom_left = (bbox[0], bbox[1] + bbox[3])
                bbox_bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])


                # Check intersection
                intersects = (bbox_top_left[0] < crop_bottom_right[0] and
                              bbox_bottom_right[0] > crop_top_left[0] and
                              bbox_top_left[1] < crop_bottom_right[1] and
                              bbox_bottom_right[1] > crop_top_left[1])
                if intersects:
                    # Adjust the bounding box to fit within the cropped region
                    clipped_x1 = max(bbox[0], x1)
                    clipped_y1 = max(bbox[1], y1)
                    clipped_x2 = min(bbox[0] + bbox[2], x2)
                    clipped_y2 = min(bbox[1] + bbox[3], y2)

                    # Convert clipped coordinates back to width/height format and adjust for new origin
                    new_bbox = [clipped_x1 - x1, clipped_y1 - y1, clipped_x2 - clipped_x1, clipped_y2 - clipped_y1]
                    bounding_boxes.append(new_bbox)
                
        return new_img, bounding_boxes






    def show_image_with_bboxes(self, index):
        """Displays both the original and cropped image with their bounding boxes side by side."""


        # Choose an image and get its ID and filename
        img_id = self.ids[index]
        img = self.coco.loadImgs([img_id])[0]

        # Construct the image path
        image_path = os.path.join(self.img_dir, img["file_name"])

        # Check if the image exists in the main directory. If not, change the path to look inside the 'person' subfolder
        if not os.path.exists(image_path):
            # Modify the path to look inside the 'person' subfolder
            image_path = os.path.join(self.img_dir, 'person', img["file_name"])

        # Open the image using PIL
        I_original = Image.open(image_path).convert('RGB')

        # Convert the PIL Image to a tensor
        to_tensor = transforms.ToTensor()
        I_cropped = to_tensor(I_original)

        # Crop the image
        I_cropped, bounding_boxes = self.crop_image(I_cropped, self.coco.loadAnns(self.coco.getAnnIds(imgIds=img["id"])))

        # Apply the transformation
        if self.transform is not None:
            I_cropped = self.transform(I_cropped)

        # Convert tensor back to PIL Image for visualization
        if isinstance(I_cropped, torch.Tensor):
            I_cropped = transforms.ToPILImage()(I_cropped)

        # Create subplots to display both images side by side
        fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

        # Helper function to draw grid
        def draw_grid(ax, width, height):
            for i in range(1, S):
                ax.axvline(x=i * width / S, color='blue', linewidth=0.2)
                ax.axhline(y=i * height / S, color='blue', linewidth=0.2)

        # Display the original image with original bounding boxes
        axarr[0].imshow(I_original)
        draw_grid(axarr[0], I_original.width, I_original.height)
        
        annIds_original = self.coco.getAnnIds(imgIds=img["id"])
        anns_original = self.coco.loadAnns(annIds_original)
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]

        for ann in anns_original:
            if ann['category_id'] == person_cat_id and 'bbox' in ann:
                bbox = ann['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
                axarr[0].add_patch(rect)
        axarr[0].set_title('Original Image')
        axarr[0].axis('off')

        # Display the cropped image with adjusted bounding boxes
        axarr[1].imshow(I_cropped)
        draw_grid(axarr[1], I_cropped.width, I_cropped.height)

        for bbox in bounding_boxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            axarr[1].add_patch(rect)
        axarr[1].set_title('Cropped Image')
        axarr[1].axis('off')

        plt.tight_layout()
        plt.show()


    

def compute_iou(bbox, cell_bbox):
    # Given a bounding box and a cell, compute the IoU

    # Convert bounding boxes to [xmin, ymin, xmax, ymax] format
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    # Calculate intersection coordinates
    inter_x1 = max(bbox[0], cell_bbox[0])
    inter_y1 = max(bbox[1], cell_bbox[1])
    inter_x2 = min(bbox[2], cell_bbox[2])
    inter_y2 = min(bbox[3], cell_bbox[3])

    # Calculate intersection area
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    
    # Calculate union area
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    cell_area = (cell_bbox[2] - cell_bbox[0]) * (cell_bbox[3] - cell_bbox[1])
    union_area = bbox_area + cell_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area

    return iou









# TO ShOW LABELS FORMAT

'''# Create an instance of the DataSetCoco class for the TRAIN dataset
coco_data = DataSetCoco(DataSetType.TRAIN, transform=None, save_crop=True)

# Fetch a sample by its index
index_to_test = 3 # You can change this to any valid index
img, yolo_targets = coco_data.__getitem__(index_to_test)

# print image name:
print("Image name:", coco_data.coco.loadImgs(coco_data.ids[index_to_test])[0]['file_name'])
print("Bounding Boxes in YOLO format:", yolo_targets)

coco_data.show_image_with_bboxes(index_to_test)
'''



