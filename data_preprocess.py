from yolo_v1 import Yolo_v1
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models 
from dataset import DataSetCoco, DataSetType
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
import json

from dataset import TRAIN, VALIDATION

def process_data():

    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256), 
            transforms.ToTensor()
        ]),
        VALIDATION: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256), 
            transforms.ToTensor()
        ]),
    }

    # Only use 'person' directory for the ImageFolder (As we do not need the other classes)
    image_datasets = {x: datasets.ImageFolder(x, data_transforms[x])
                    for x in [TRAIN, VALIDATION]}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                for x in [TRAIN, VALIDATION]}



    # Printing the classes for verification
    print("Classes in TRAIN dataset:", image_datasets[TRAIN].classes)
    print("Classes in VALIDATION dataset:", image_datasets[VALIDATION].classes)

    return dataloaders, image_datasets



    '''data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    VALIDATION: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }'''



def move_images_with_persons_to_person_dir(coco_dataset):
    img_dir = coco_dataset.img_dir  # Image directory
    ann_file = coco_dataset.ann_file  # Annotation JSON file

    person_img_dir = os.path.join(img_dir, 'person')  # Person image directory

    # Ensure the 'person' directory exists, if not, create it
    if not os.path.exists(person_img_dir):
        os.makedirs(person_img_dir)

    # Read the COCO annotation JSON file for the given dataset type
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # Extract category IDs for the 'person' class
    person_cat_id = [cat['id'] for cat in data['categories'] if cat['name'] == 'person'][0]

    # Get image ids of images containing persons
    img_ids_with_persons = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] == person_cat_id]

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



#coco_set_val = DataSetCoco(DataSetType.VALIDATION)
#move_images_with_persons_to_person_dir(coco_set_val)
#coco_set_val = DataSetCoco(DataSetType.VALIDATION)
#move_images_with_persons_to_person_dir(coco_set_val)


process_data()