import csv
from yolo_v1 import Yolo_v1
from loss import YOLOLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset import TRAIN, VALIDATION
from yolo_v1 import Yolo_v1 
from model_constants import S, B, C
import copy
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType




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

    iou = inter_area / (union_area + 1e-6) # Add a small epsilon to prevent division by zero
    return iou



def compute_accuracy(preds, labels):
    correct = 0
    localization = 0
    similar = 0  # This variable is not used in your original function; consider its purpose.
    other = 0
    background = 0
    
    batch_size = preds.shape[0] # Predictions are of shape (batch_size, S, S, B*5 + C)
    
    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                # Extract bounding box and class prediction
                pred_bbox1 = preds[b, i, j, 2:6].tolist() # (x1, y1, w1, h1)
                pred_bbox2 = preds[b, i, j, 7:11].tolist() # (x2, y2, w2, h2)
                pred_class_prob = preds[b, i, j, 0].item()

                label_bbox = labels[b, i, j, 2:6].tolist() # (x, y, w, h)
                label_class_prob = labels[b, i, j, 0].item() # Object presence probability

                # Only consider cells where the label indicates there is an object
                if label_class_prob == 1:
                    # Find the predicted bbox with the highest IoU
                    iou1 = bbox_iou(pred_bbox1, label_bbox)
                    iou2 = bbox_iou(pred_bbox2, label_bbox)
                    iou = max(iou1, iou2)
                    
                    # Check the conditions for scoring
                    if pred_class_prob > 0.5 and iou > 0.5:
                        correct += 1
                    elif pred_class_prob > 0.5 and 0.1 < iou <= 0.5:
                        localization += 1
                    elif pred_class_prob > 0.5 and iou <= 0.1:
                        background += 1
                    elif iou > 0.1:
                        other += 1  # All other cases as "other"

    return correct, localization, similar, other, background








def plot_training_results(losses, metrics, num_epochs):
    plt.figure(figsize=(12,5))
    
    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(losses[TRAIN], label="Training Loss")
    plt.plot(losses[VALIDATION], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Running Loss Over Epochs")
    plt.legend()
    
    # Plotting pie chart for error distribution
    last_epoch_metrics = metrics[VALIDATION][num_epochs-1]
    labels = last_epoch_metrics.keys()
    sizes = [last_epoch_metrics[k] for k in labels]
    
    plt.subplot(1, 2, 2)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Error Distribution in Last Epoch")
    
    plt.tight_layout()
    plt.show()




    





def train(model: Yolo_v1, criterion: YOLOLoss, optimizer, scheduler=None, num_epochs=4):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    image_datasets = {
        #TRAIN: DataSetCoco(DataSetType.TRAIN, transform=data_transforms[TRAIN]),
        #VALIDATION: DataSetCoco(DataSetType.VALIDATION, transform=data_transforms[VALIDATION])
        TRAIN: DataSetCoco(DataSetType.TRAIN, training=True),
        VALIDATION: DataSetCoco(DataSetType.VALIDATION, training=False)
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
                   for x in [TRAIN, VALIDATION]}

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALIDATION]}

    # Dictionaries to store metrics for each epoch
    losses = {TRAIN: [], VALIDATION: []}
    metrics = {TRAIN: {}, VALIDATION: {}}


    # Open the CSV file once before you start the epochs.
    csv_file = open("training_metrics.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Train_Loss", "Validation_Loss"]) # Write the header row.


    
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
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() # * inputs.size(0) is removed because YOLOLoss already sums over the batch

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print(f"{phase} Correct Predictions: {all_correct}")
            print(f"{phase} Localization Errors: {all_localization}")
            print(f"{phase} Similar Object Errors: {all_similar}")
            print(f"{phase} Other Errors: {all_other}")
            print(f"{phase} Background Predictions: {all_background}")

            # Save the metrics for this epoch
            losses[phase].append(epoch_loss)
            metrics[phase][epoch] = {
                "correct": all_correct,
                "localization": all_localization,
                "similar": all_similar,
                "other": all_other,
                "background": all_background
            }
            
            epoch_acc = all_correct / dataset_sizes[phase]

            # Deep copy the model
            if phase == VALIDATION and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



        writer.writerow([epoch, losses[TRAIN][-1], losses[VALIDATION][-1]])  # write the most recent losses after each epoch




    csv_file.close()
    torch.save(best_model_wts, "best_model_weights.pth") # Saving the best model


    # Plotting the training results
    plot_training_results(losses, metrics, num_epochs)





def main():  # Encapsulating in main function
    
    print("DEVICE:", DEVICE)

    model = Yolo_v1()
    model = model.to(DEVICE)  # Use GPU if available

    criterion = YOLOLoss()  # Loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Observe that all parameters are being optimized

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs

    # Start training process
    model = train(model, criterion, optimizer, num_epochs=25)

# The following is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()


