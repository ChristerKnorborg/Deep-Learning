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

from dataset import TRAIN, VALIDATION

def process_data():
    train_dataset = DataSetCoco(DataSetType.TRAIN)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)



    data_transforms = {
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
    }

    data_dir = 'data'
    print(os.path.join(data_dir, TRAIN))
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                data_transforms[x])
        for x in [TRAIN, VALIDATION]}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in [TRAIN, VALIDATION]}

    return dataloaders, image_datasets
            