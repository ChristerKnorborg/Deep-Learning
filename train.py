from yolo_v1 import Yolo_v1
from loss import YOLOLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_preprocess import process_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset import TRAIN, VALIDATION
from yolo_v1 import Yolo_v1 
from model_constants import S, B, C


DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available













def xywh_to_x1y1x2y2(box):
    """
    Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    S: grid size (e.g., 7 for a 7x7 grid).
    """
    x_center, y_center, width, height = box

    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2

    return [x1, y1, x2, y2]


def bbox_iou(prediction_box, label_box):
    """
    Returns the IoU of two bounding boxes.
    """


    b1_x1, b1_y1, b1_x2, b1_y2 = xywh_to_x1y1x2y2(prediction_box)
    b2_x1, b2_y1, b2_x2, b2_y2 = xywh_to_x1y1x2y2(label_box)

    # Compute the coordinates of the intersection rectangle
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # Area of intersection
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    # Area of union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    iou = inter_area / union_area
    return iou



def compute_accuracy(preds, labels):
    correct = 0
    localization = 0
    similar = 0
    other = 0
    background = 0
    
    batch_size = preds.shape[0]
    
    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                # Extract bounding box and class prediction
                pred_bbox1 = preds[b, i, j, 1:5].tolist()
                pred_bbox2 = preds[b, i, j, 6:10].tolist()
                pred_class_prob = preds[b, i, j, 0].item()

                label_bbox = labels[b, i, j, 2:6].tolist()
                label_class_prob = labels[b, i, j, 0].item()

                # Find the predicted bbox with the highest IoU
                iou1 = bbox_iou(pred_bbox1, label_bbox)
                iou2 = bbox_iou(pred_bbox2, label_bbox)
                iou = max(iou1, iou2)
                
                # Check the conditions for scoring
                if pred_class_prob > 0.5 and iou > 0.5:
                    correct += 1
                elif pred_class_prob > 0.5 and 0.1 < iou < 0.5:
                    localization += 1
                elif pred_class_prob > 0.5 and iou < 0.1:
                    background += 1
                elif 0.1 < iou:
                    other += 1  # As we treat all other cases as "other"
                    
    return correct, localization, similar, other, background







def train(model: Yolo_v1, criterion, optimizer, scheduler, num_epochs=25):

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataloaders, image_datasets = process_data()
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALIDATION]}

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')
        

            


        for phase in [TRAIN, VALIDATION]:
            
            if phase == TRAIN:
                print('Training phase:')
                model.train()
            else:
                print('Validation phase:')
                model.eval()
            
            # Initialize to 0. E.g. running_corrects = 0 will not work due to type mismatch in double
            # running_corrects: torch.Tensor = torch.tensor(0)
            running_loss = 0.0

            all_correct = 0
            all_localization = 0
            all_similar = 0
            all_other = 0
            all_background = 0

            # Print progress within an epoch
            total_batches = len(dataloaders[phase]) 

            for iteration, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                print(f"\rProcessing batch {iteration}/{total_batches}", end="")

                inputs = inputs.to(DEVICE) 
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == TRAIN):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                   
                    # Compute accuracy metrics
                    correct, localization, similar, other, background = compute_accuracy(outputs, labels)
                    all_correct += correct
                    all_localization += localization
                    all_similar += similar
                    all_other += other
                    all_background += background

                    if phase == TRAIN:
                        loss.backward() # Backpropagation (luckily, PyTorch does this automatically for us)
                        optimizer.step()
                        #scheduler.step() # Decay learning rate by a factor of 0.1 every 7 epochs (Comes after optimizer)
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print(f"{phase} Correct Predictions: {all_correct}")
            print(f"{phase} Localization Errors: {all_localization}")
            print(f"{phase} Similar Object Errors: {all_similar}")
            print(f"{phase} Other Errors: {all_other}")
            print(f"{phase} Background Predictions: {all_background}")

            # Here we will use all_correct as the metric for determining the best model. 
            # This can be adjusted depending on the needs.
            epoch_acc = all_correct / dataset_sizes[phase]

            # Deep copy the model
            if phase == VALIDATION and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Load the best model's weights and return
    model.load_state_dict(best_model_wts)
    return model




model = Yolo_v1()
model = model.to(DEVICE) # Use GPU if available

criterion = YOLOLoss() # Loss function
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9) # Observe that all parameters are being optimized
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs

model = train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)