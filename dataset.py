# Import grid dimension S = 7, Bounding boxes per cell B = 2, and classes C = 1
from model_constants import S, B
import copy
import json
import os
from random import random
import shutil
from torchvision import transforms
from typing import Any
import requests
import zipfile
from pycocotools.coco import COCO
import torchvision.transforms.functional as F

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
TEST = "./data/test2017"


# Define the DataSetType enum
class DataSetType(Enum):
    TRAIN = "./data/train2017", "./data/labels/annotations/instances_train2017.json"
    VALIDATION = "./data/val2017", "./data/labels/annotations/instances_val2017.json"
    TEST = "./data/test2017", "./data/labels/annotations/instances_test2017.json"


class DataSetCoco(Dataset):

    def __init__(self, datatype: DataSetType, subset_size=None, training=False, save_augmentation=False, chosen_images=None):
        """Initializes the dataset. Downloads and extracts data if needed.

        Args:
        - datatype (DataSetType): The type of dataset to use consisting of:
            - img_dir (str): Path to the directory containing images.
            - ann_file (str): Path to the annotation file.
        - transform (callable, optional): Optional transform to be applied 
        """

        # ... your initialization code ...

        self.img_dir, self.ann_file = datatype.value
        self._ensure_data_exists_or_download()  # Ensure data exists or download it
        self.coco = COCO(self.ann_file)
        # Move images with persons to the 'person' directory (if not already done)
        self.move_images_with_persons_to_person_dir

        self.training = training

        person_category_ids = self.coco.getCatIds(catNms=['person'])
        self.ids = self.coco.getImgIds(catIds=person_category_ids)

        # self.ids = list(self.coco.imgs.keys())

        self.chosen_images = chosen_images
        if self.chosen_images != None:
            # Fetch all images details from the COCO dataset
            all_images = self.coco.loadImgs(ids=self.coco.getImgIds())

            # Extract the image IDs that match the filenames
            image_ids = [img['id']
                         for img in all_images if img['file_name'] in self.chosen_images]
            chosen_ids = image_ids
            self.ids = [img_id for img_id in self.ids if img_id in chosen_ids]

        # Reduce the dataset size if subset_size is set
        if subset_size is not None and subset_size < len(self.ids):
            self.ids = random.sample(self.ids, subset_size)

        # If True, saves the last data augmentation for picture plotting (e.g. crop_image, color_image, etc.)
        self.save_augmentation = save_augmentation
        self.augmented_records = {
            'image': None,  # The transformed image tensor
            'bboxes': None,  # The transformed bounding boxes
        }  # Used to store the data augmentation for an image if save_augmentation is True


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

        person_img_dir = os.path.join(
            img_dir, 'person')  # Person image directory

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


    def get_all_image_file_names(self):
        """
        Retrieve the file names of all images in the dataset.

        Returns:
        - List[str]: The file names of all images.
        """
        all_image_ids = self.ids  # IDs of all images
        all_image_infos = self.coco.loadImgs(
            all_image_ids)  # Load all image info from COCO
        all_file_names = [img_info['file_name']
                          for img_info in all_image_infos]  # Extract file names
        return all_file_names

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
        # Changed from len(self.coco.imgs) to len(self.ids) to account for the fact that we're only using images with persons in them
        return len(self.ids)

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

        img = Image.open(image_path).convert('RGB')  # Open the image using PIL
        img = transforms.ToTensor()(img)  # Convert the PIL Image to a tensor

        ### THIS NEEDS TO BE APPLIED TO MAKE THE ENCODER WORK FOR BOTH TRAINING AND VALIDATION. It makes width and height the same ###
        # Crop the image and adjust bounding boxes accordingly
        img, bounding_boxes = self.crop_image(img, annotations)
        img, bounding_boxes = self.resize_image(img, bounding_boxes) # Resize to make dataloader work with tensor as all images need to be the same size in batches
        bounding_boxes = self.remove_small_bounding_boxes(bounding_boxes) # Remove small bounding boxes

        # If training, apply data augmentation
        if self.training:
            img = self.color_image(img) # Apply color augmentation
            img, bounding_boxes = self.horizontal_flip_image(img, bounding_boxes) # Apply horizontal flip with 0.5 probability

        # Get the width and height (after transforming the image if training)
        img_height, img_width = img.shape[1:3]

        # The image is divided into a grid of size S x S. with each cell being of size 5 (confidence, x, y, w, h)
        label_tensor = torch.zeros((S, S, 5))

        # Convert COCO bounding boxes to YOLO format
        for bbox in bounding_boxes:

            # Find midpoint coordinate (x, y) of bounding box. Notice bbox is in format [x1, y1, width, height]
            x_center = bbox[0] + (bbox[2] / 2)  # Center is x1 + half of width
            y_center = bbox[1] + (bbox[3] / 2)  # Center is y1 + half of height

            # Normalize the coordinates
            x = x_center / img_width
            y = y_center / img_height

            # Normalize the width and height of the bounding box (relative to the image size)
            w = bbox[2] / img_width  # bbox[2] is the width of the bounding box
            # bbox[3] is the height of the bounding box
            h = bbox[3] / img_height

            # Determine grid cell
            i, j = int(y * S), int(x * S)
            x_cell_offset = x*S - j  # Offset of midpoint x coordinate from the left side of the cell
            y_cell_offset = y*S - i  # Offset of midpoint y coordinate from the top side of the cell

            # Update cell values with the formula (P_c, x, y, w, h), where P_c is the probability of a person object being present in
            # the cell (Combined Objectness/class score), and (x, y) are the coordinates of the midpoint of the bounding box relative
            # to its grid cell, and (w, h) are the width and height of the bounding box relative to the whole image.

            if label_tensor[i, j, 0] == 0:  # If there's no object in the cell yet

                # Class score for "person" (Objectness/class score of the object being a "person" is true)
                label_tensor[i, j, 0] = 1

                # Set bounding box attributes
                label_tensor[i, j, 1:5] = torch.tensor(
                    [x_cell_offset, y_cell_offset, w, h])

            else:  # Choose the bounding box with the largest area

                # Get the current bounding box attributes
                current_w = label_tensor[i, j, 3]
                current_h = label_tensor[i, j, 4]

                # Calculate the area of the current and new bounding box
                current_area = current_w * current_h
                new_area = w * h

                # If the new bounding box has a larger area, replace the old bounding box with the new one
                if new_area > current_area:

                    # Class score for "person" (Objectness/class score of the object being a "person" is true)
                    label_tensor[i, j, 0] = 1

                    # Set bounding box attributes
                    label_tensor[i, j, 1:5] = torch.tensor(
                        [x_cell_offset, y_cell_offset, w, h])

        # Save the data augmentation if save_augmentation is True
        if self.save_augmentation:
            self.augmented_records = {
                'image': img,  # The transformed image tensor
                'bboxes': bounding_boxes,  # The transformed bounding boxes
            }

        return img, label_tensor, img_id

    def crop_image(self, img: torch.Tensor, original_annotations, size=None):

        if size is None:
            original_height, original_width = img.shape[1:3]
            # Chose to use original width or original height randomly for new size
            chosen_dim = random.choice([original_height, original_width])
            # Calculate a random scale factor between 0.8 and 1.2
            scale_factor = random.uniform(0.8, 1.2)

            # Calculate the new size
            new_size = (int(chosen_dim * scale_factor),
                        int(chosen_dim * scale_factor))
            # Determine crop dimensions
            new_width, new_height = new_size
        else:
            new_width, new_height = size

        # If the image dimensions are smaller than crop dimensions, pad the image
        padding_top = padding_bottom = padding_left = padding_right = 0
        if img.shape[1] < new_height or img.shape[2] < new_width:
            padding_top = max(0, (new_height - img.shape[1]) // 2)
            padding_bottom = max(0, new_height - img.shape[1] - padding_top)
            padding_left = max(0, (new_width - img.shape[2]) // 2)
            padding_right = max(0, new_width - img.shape[2] - padding_left)

            img = torch.nn.functional.pad(
                img, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)

        # Randomly choose a top-left corner for cropping
        max_x = img.shape[2] - new_width
        max_y = img.shape[1] - new_height
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        x2 = x1 + new_width
        y2 = y1 + new_height

        # Crop the image
        new_img = img[:, y1:y2, x1:x2]

        # Adjust the annotations and bounding boxes
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]
        bounding_boxes = []
        processed_annotations = []
        for ann in original_annotations:
            new_ann = copy.deepcopy(
                ann) if ann['category_id'] == person_cat_id else ann

            if new_ann['category_id'] == person_cat_id:
                # Adjust 'bbox' for padding if necessary
                new_ann['bbox'][0] += padding_left
                new_ann['bbox'][1] += padding_top

                # Define the corners for the bounding box
                bbox = new_ann['bbox']
                bbox_top_left = (bbox[0], bbox[1])
                bbox_bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])

                # Check intersection with crop area
                intersects = (bbox_top_left[0] < x2 and bbox_bottom_right[0] > x1 and
                              bbox_top_left[1] < y2 and bbox_bottom_right[1] > y1)

                if intersects:
                    # Adjust the bounding box to fit within the cropped region
                    clipped_x1 = max(bbox[0], x1)
                    clipped_y1 = max(bbox[1], y1)
                    clipped_x2 = min(bbox[0] + bbox[2], x2)
                    clipped_y2 = min(bbox[1] + bbox[3], y2)

                    # Convert clipped coordinates back to width/height format and adjust for new origin
                    new_bbox = [clipped_x1 - x1, clipped_y1 - y1,
                                clipped_x2 - clipped_x1, clipped_y2 - clipped_y1]
                    bounding_boxes.append(new_bbox)

            processed_annotations.append(new_ann)

        # Continue with your process, using new_img and processed_annotations as needed
        return new_img, bounding_boxes

    def color_image(self, img):
        """
        Apply color augmentation to the image.

        Args:
        img (PIL Image or Tensor): Image to be augmented.

        Returns:
        Tensor: Augmented image.
        """
        color_transforms = transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.1, 0.1)
        )
        img = color_transforms(img)
        return img

    def resize_image(self, img: torch.Tensor, bounding_boxes, size=(512, 512)):
        """
        Resize the image randomly between 0.8 and 1.2 times its original size and adjust the bounding boxes accordingly.

        Args:
        img (PIL Image or Tensor): Image to be resized.
        bounding_boxes (list): List of bounding boxes present in the image.

        Returns:
        Tuple[Tensor, list]: Resized image and the adjusted list of bounding boxes.
        """
        # Store original dimensions for scaling calculation
        original_img_height, original_img_width = img.shape[1:3]

        # New dimensions
        new_width, new_height = size

        # Resize the image to the new dimensions
        img = transforms.Resize((new_height, new_width),
                                antialias=True)(img)  # type: ignore

        # Calculate scaling factors for the bounding boxes
        width_scale = new_width / original_img_width
        height_scale = new_height / original_img_height

        # Adjust bounding boxes. The bounding boxes are expected in format [x1, y1, x2, y2]
        adjusted_bounding_boxes = []
        for bbox in bounding_boxes:
            adjusted_bbox = [
                bbox[0] * width_scale,  # x1
                bbox[1] * height_scale,  # y1
                bbox[2] * width_scale,  # x2
                bbox[3] * height_scale,  # y2
            ]
            adjusted_bounding_boxes.append(adjusted_bbox)

        return img, adjusted_bounding_boxes

    def horizontal_flip_image(self, img, bounding_boxes):
        """
        Randomly flip the image horizontally with a probability of 0.5 and adjust the bounding boxes accordingly.

        Args:
        img (Tensor): Image to be potentially flipped.
        bounding_boxes (list): List of bounding boxes present in the image.

        Returns:
        Tuple[Tensor, list]: Potentially flipped image and the adjusted list of bounding boxes.
        """

        # We need the width to adjust the bounding box coordinates
        img_width = img.shape[2]

        # Flip the image horizontally with a probability of 0.5
        flipped_img = transforms.RandomHorizontalFlip(p=1)(img)

        adjusted_bounding_boxes = []
        for bbox in bounding_boxes:
            # Calculate the new x-coordinate for the flipped image
            # new_x1 is calculated as "width - old_x1 - old_width"
            new_x1 = img_width - bbox[0] - bbox[2]

            # The bounding box dimensions remain the same, only the origin (top-left corner) changes.
            # (x1, y1, width, height) format
            adjusted_bbox = [new_x1, bbox[1], bbox[2], bbox[3]]

            adjusted_bounding_boxes.append(adjusted_bbox)

        return flipped_img, adjusted_bounding_boxes
    
    

    def remove_small_bounding_boxes(self, bounding_boxes, min_size=25):
        """
        Filters out bounding boxes that are below a certain size threshold.

        Args:
        bounding_boxes (list): List of bounding boxes, each represented as [x1, y1, width, height].
        min_size (int): The minimum acceptable width and height of a bounding box.

        Returns:
        list: The filtered list of bounding boxes.
        """
        filtered_boxes = []
        for bbox in bounding_boxes:
            # Extracting width and height from the box representation
            width, height = bbox[2], bbox[3]

            # Boxes must be at least min_size in both width and height
            if width >= min_size and height >= min_size:
                filtered_boxes.append(bbox)

        return filtered_boxes

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
        original_image = Image.open(image_path).convert('RGB')

        if not self.save_augmentation:
            print(
                "You need to set save_augmentation to True to be able to visualize the data augmentation.")
            return

        # Ensure that there's an augmented image to display
        if self.augmented_records['image'] is None or self.augmented_records['bboxes'] is None:
            print(
                "No augmented image to display. Please make sure the augmentation was applied.")
            return

        # Retrieve the augmented image and bounding boxes
        augmented_image = self.augmented_records['image']
        bounding_boxes = self.augmented_records['bboxes']

        # Convert tensor back to PIL Image for visualization
        if isinstance(augmented_image, torch.Tensor):
            augmented_image = transforms.ToPILImage()(augmented_image)

        # Retrieve dimensions for the augmented image
        augmented_width, augmented_height = augmented_image.size

        # Create subplots to display both images side by side
        fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

        # Helper function to draw grid
        def draw_grid(ax, width, height):
            for i in range(1, S):
                ax.axvline(x=i * width / S, color='blue', linewidth=0.2)
                ax.axhline(y=i * height / S, color='blue', linewidth=0.2)

        # Display the original image with original bounding boxes
        axarr[0].imshow(original_image)
        # unpacking the size tuple directly
        draw_grid(axarr[0], *original_image.size)

        annIds_original = self.coco.getAnnIds(imgIds=img["id"])
        anns_original = self.coco.loadAnns(annIds_original)
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]

        for ann in anns_original:
            if ann['category_id'] == person_cat_id and 'bbox' in ann:
                bbox = ann['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
                axarr[0].add_patch(rect)

        axarr[0].set_title('Original Image')
        axarr[0].axis('off')

        # Display the cropped image with adjusted bounding boxes
        axarr[1].imshow(augmented_image)
        draw_grid(axarr[1], augmented_width, augmented_height)

        for bbox in bounding_boxes:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            axarr[1].add_patch(rect)

        axarr[1].set_title('Augmented Image')
        axarr[1].axis('off')

        plt.tight_layout()
        plt.show()

    def get_image_id_from_filename(self, filename):
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            img_info = self.coco.loadImgs([img_id])[0]
            if img_info["file_name"] == filename:
                return img_info["id"]
        return 0


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
coco_data = DataSetCoco(DataSetType.TRAIN, save_augmentation=True, training=True)

# Fetch a sample by its index
index_to_test = 1 # You can change this to any valid index
img, yolo_targets, _ = coco_data.__getitem__(index_to_test)

# print image name:
print("Image name:", coco_data.coco.loadImgs(coco_data.ids[index_to_test])[0]['file_name'])
print("Bounding Boxes in YOLO format:", yolo_targets)

coco_data.show_image_with_bboxes(index_to_test)
'''