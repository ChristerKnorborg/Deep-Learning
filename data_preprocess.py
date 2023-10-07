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
            # First arguments for inital trainings
            # transforms.Resize(256),
            # transforms.CenterCrop(256),
            
            transforms.RandomResizedCrop(256),
            # Horizontally flip the image with probability 0.5
            transforms.RandomHorizontalFlip(0.5),
            # Randomly change the brightness of the image by 10%
            transforms.ColorJitter(brightness=0.1),
            # Randomly rotate images in the range (degrees, 0 to 180)
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        VALIDATION: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

   # Use our custom DataSetCoco class
    image_datasets = {
        TRAIN: DataSetCoco(DataSetType.TRAIN, transform=data_transforms[TRAIN]),
        VALIDATION: DataSetCoco(DataSetType.VALIDATION, transform=data_transforms[VALIDATION])
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                   for x in [TRAIN, VALIDATION]}

    # Printing the classes for verification
    # Note: This will print only one class ('person') since that's what pur dataset contains
    print("Classes in TRAIN dataset:", ["person"])
    print("Classes in VALIDATION dataset:", ["person"])

    return dataloaders, image_datasets



# process_data()
