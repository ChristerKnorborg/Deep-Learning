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

   # Use our custom DataSetCoco class
    image_datasets = {
        #TRAIN: DataSetCoco(DataSetType.TRAIN, transform=data_transforms[TRAIN]),
        #VALIDATION: DataSetCoco(DataSetType.VALIDATION, transform=data_transforms[VALIDATION])
        TRAIN: DataSetCoco(DataSetType.TRAIN, subset_size=10000, training=True),
        VALIDATION: DataSetCoco(DataSetType.VALIDATION, subset_size=1000)
    }


    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in [TRAIN, VALIDATION]}

    # Printing the classes for verification
    # Note: This will print only one class ('person') since that's what pur dataset contains
    print("Classes in TRAIN dataset:", ["person"])
    print("Classes in VALIDATION dataset:", ["person"])

    return dataloaders, image_datasets



# process_data()
