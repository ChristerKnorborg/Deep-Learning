from yolo_v1 import Yolo_v1
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models 
from dataset import DataSetCoco, DataSetType
from torch.optim import lr_scheduler
from data_preprocess import process_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset import TRAIN, VALIDATION

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU if available




dataloaders, image_datasets = process_data()
dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALIDATION]}

def train(model, criterion, optimizer, scheduler, num_epochs = 25):

    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')
        
        for phase in [TRAIN, VALIDATION]:
            if phase == TRAIN:
                #scheduler.step()
                model.train()
            else:
                model.eval()

            # Initialize to 0. E.g. running_corrects = 0 will not work due to type mismatch in double
            running_corrects: torch.Tensor = torch.tensor(0) 
            running_loss = 0.0

            print("Phase: ", len(dataloaders[phase]))
            print("Phase:", dataloaders[phase])
            i = 0
            for inputs, labels in dataloaders[phase]:
                i = i + 1
                if i == 10:
                    break
                
                print(inputs.shape)  # This should print something like [batch_size, 3, height, width]
                print(labels.shape)  # This should print [batch_size] if you're doing classification.
                inputs = inputs.to(DEVICE) 
                labels = labels.to(DEVICE)
                
                #optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == TRAIN:
                        loss.backward() # Backpropagation (luckily, PyTorch does this automatically for us)
                        #optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == VALIDATION and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    return model
        

model = Yolo_v1()

# Freeze all layers (i.e., disable training)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layer (named fc)
for param in model.fc.parameters():
    param.requires_grad = True

# Put the model on the GPU
model = model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)


