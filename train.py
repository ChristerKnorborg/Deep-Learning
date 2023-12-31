import csv
import datetime
from yolo_network import Yolo_network
from loss import YOLOLoss
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import copy
from dataset import TRAIN, VALIDATION
from yolo_network import Yolo_network
from model_constants import S, DEVICE
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType


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

    # Add a small epsilon to prevent division by zero
    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_accuracy(preds, labels):

    # TP Means the label is good, the class is good, and the IOU is good
    TP_class_loca_good = 0
    # FP means the label is bad
    FP_class_good_IOU_bad = 0
    FP_class_bad_IOU_good = 0
    FP_class_IOU_good = 0
    # FN means the label is good, but the prediction does not choose it
    FN_class_good_IOU_bad = 0
    FN_class_bad_IOU_good = 0
    FN_class_bad_IOU_bad = 0

    correct = 0
    localization = 0
    other = 0
    otherbighalf = 0
    background = 0
    total_bounding_boxes = 0

    # Predictions are of shape (batch_size, S, S, B*5)
    batch_size = preds.shape[0]

    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                # Extract bounding box and class prediction
                pred_bbox1 = preds[b, i, j, 1:5].tolist()  # (x1, y1, w1, h1)
                pred_bbox2 = preds[b, i, j, 6:10].tolist()  # (x2, y2, w2, h2)
                pred_person_prob = preds[b, i, j, 0].item()

                label_bbox = labels[b, i, j, 1:5].tolist()  # (x, y, w, h)
                # Person presence probability
                label_person_prob = labels[b, i, j, 0].item()

                # Only consider cells where the label indicates there is an object
                # if the label is 1. IE person class in grid and the prediction class is > 0.5 and localization IOU > 0.5
                if label_person_prob == 1:
                    iou1 = bbox_iou(pred_bbox1, label_bbox)
                    iou2 = bbox_iou(pred_bbox2, label_bbox)
                    iou = max(iou1, iou2)

                    if pred_person_prob >= 0.5 and iou >= 0.5:
                        TP_class_loca_good += 1
                    if pred_person_prob < 0.5 and iou >= 0.5:
                        FN_class_bad_IOU_good += 1
                    if pred_person_prob >= 0.5 and iou < 0.5:
                        FN_class_good_IOU_bad += 1
                    if pred_person_prob < 0.5 and iou < 0.5:
                        FN_class_bad_IOU_bad += 1

                if label_person_prob != 1:
                    iou1 = bbox_iou(pred_bbox1, label_bbox)
                    iou2 = bbox_iou(pred_bbox2, label_bbox)
                    iou = max(iou1, iou2)
                    if pred_person_prob >= 0.5 and iou >= 0.5:
                        FP_class_IOU_good += 1
                    if pred_person_prob < 0.5 and iou >= 0.5:
                        FP_class_bad_IOU_good += 1
                    if pred_person_prob >= 0.5 and iou < 0.5:
                        FP_class_good_IOU_bad += 1

                if label_person_prob == 1:
                    # Find the predicted bbox with the highest IoU
                    iou1 = bbox_iou(pred_bbox1, label_bbox)
                    iou2 = bbox_iou(pred_bbox2, label_bbox)
                    iou = max(iou1, iou2)

                    # Check the conditions for scoring
                    if pred_person_prob > 0.5 and iou > 0.5:
                        correct += 1
                    elif pred_person_prob > 0.5 and 0.1 < iou <= 0.5:
                        localization += 1
                    elif pred_person_prob > 0.5 and iou <= 0.1:
                        background += 1
                    elif iou > 0.1:
                        if iou <= 0.5:
                            other += 1  # All other cases as "other"
                        if iou > 0.5:
                            otherbighalf += 1

                    total_bounding_boxes += 1  # Only count the bounding boxes where there is a person

    return correct, localization, other, background, total_bounding_boxes, otherbighalf, TP_class_loca_good, FP_class_bad_IOU_good, FP_class_good_IOU_bad, FP_class_IOU_good, FN_class_good_IOU_bad, FN_class_bad_IOU_good, FN_class_bad_IOU_bad


def plot_training_results(losses, metrics, num_epochs):
    plt.figure(figsize=(12, 5))

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


def save_model(model_weights, num_epochs, optimizer, scheduler=None, subset_size=None, batch_size=None, save_dir='models'):
    """
    Save the model with a filename that encapsulates various training parameters.

    :param model_weights: State dictionary of the model.
    :param num_epochs: Total number of epochs in training.
    :param optimizer: Optimizer used in training.
    :param scheduler: Learning rate scheduler used in training. Can be None.
    :param subset_size: The size of the dataset subset used in training. Can be None.
    :param batch_size: The size of the batches used in training. Can be None.
    :param save_dir: Directory to save the model.
    """
    # Ensure the save directory exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract training parameters from the optimizer.
    # Assuming there is one learning rate for all parameter groups.
    learning_rate = optimizer.param_groups[0]['lr']

    # Extract scheduler parameters if a scheduler is provided.
    if scheduler:
        step_size = scheduler.step_size if hasattr(
            scheduler, 'step_size') else "unknown"
        gamma = scheduler.gamma if hasattr(scheduler, 'gamma') else "unknown"
    else:
        step_size = "none"
        gamma = "none"

    # Current date and time for filename.
    date_now = datetime.datetime.now().strftime('%m-%d_%H')

    # Construct the unique filename. Including subset_size and batch_size in the file name.
    model_filename = f"model_{date_now}_epoch-{num_epochs}_LR-{learning_rate}_step-{step_size}_gamma-{gamma}_subset-{subset_size}_batch-{batch_size}.pth"
    model_path = os.path.join(save_dir, model_filename)

    # Save the model weights.
    torch.save(model_weights, model_path)


def train(model: Yolo_network, criterion: YOLOLoss, optimizer, scheduler=None, num_epochs=25, fine_tuning_epochs=0, chosen_train_images=None):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    SUBSET_SIZE = 100
    BATCH_SIZE = 10
    VALIDATION_SIZE = 20  # 2000 images for validation

    image_datasets = {
        TRAIN: DataSetCoco(DataSetType.TRAIN, training=True, subset_size=SUBSET_SIZE, chosen_images=chosen_train_images),
        VALIDATION: DataSetCoco(DataSetType.VALIDATION,
                                training=False, subset_size=VALIDATION_SIZE)
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in [TRAIN, VALIDATION]}

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALIDATION]}

    # Dictionaries to store metrics for each epoch
    losses = {TRAIN: [], VALIDATION: []}
    metrics = {TRAIN: {}, VALIDATION: {}}

    # Retrieve and print all image file names at the beginning of training
    print("Retrieving all image file names...")
    train_image_file_names = image_datasets[TRAIN].get_all_image_file_names()
    val_image_file_names = image_datasets[VALIDATION].get_all_image_file_names(
    )

    print("\nTraining Images:")
    for file_name in train_image_file_names:
        print(file_name)

    print("\nValidation Images:")
    for file_name in val_image_file_names:
        print(file_name)

    # Open the CSV file once before you start the epochs.
    csv_file = open("training_metrics.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss',
                     'Correct Predictions', 'Localization Errors',
                     'Other Errors',
                     'Background Predictions', 'Total Bounding Boxes', 'TP_class_loca_good', 'FP_class_bad_IOU_good', 'FP_class_good_IOU_bad', 'FP_class_IOU_good', 'FN_class_good_IOU_bad', 'FN_class_bad_IOU_good', 'FN_class_bad_IOU_bad'])

    # Define the fine-tuning phase start
    start_fine_tuning_epoch = num_epochs - fine_tuning_epochs

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')

        # Check if it's time to start fine-tuning
        if epoch == start_fine_tuning_epoch:
            print("Starting fine-tuning phase.")
            # Unfreeze all the layers for fine-tuning
            for param in model.parameters():
                param.requires_grad = True

            optimizer.param_groups[0]['lr'] = 0.0001 # adjust the learning rate for fine-tuning

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

            all_other = 0
            all_otherbighalf = 0
            all_background = 0
            all_bounding_boxes = 0

            # TP Means the label is good, the class is good, and the IOU is good
            all_TP_class_loca_good = 0
            # FP means the label is bad
            all_FP_class_good_IOU_bad = 0
            all_FP_class_bad_IOU_good = 0
            all_FP_class_IOU_good = 0
            # FN means the label is good, but the prediction does not choose it
            all_FN_class_good_IOU_bad = 0
            all_FN_class_bad_IOU_good = 0
            all_FN_class_bad_IOU_bad = 0

            # Print progress within an epoch
            total_batches = len(dataloaders[phase])

            for iteration, (inputs, labels, _) in enumerate(dataloaders[phase], start=1):
                print(
                    f"\rProcessing batch {iteration}/{total_batches}", end="")

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Compute accuracy metrics
                    correct, localization, other, background, bounding_boxes, otherbighalf, TP_class_loca_good, FP_class_bad_IOU_good, FP_class_good_IOU_bad, FP_class_IOU_good, FN_class_good_IOU_bad, FN_class_bad_IOU_good, FN_class_bad_IOU_bad = compute_accuracy(
                        outputs, labels)
                    all_correct += correct
                    all_localization += localization
                    all_other += other
                    all_otherbighalf += otherbighalf
                    all_background += background
                    all_bounding_boxes += bounding_boxes
                    all_TP_class_loca_good += TP_class_loca_good
                    all_FP_class_bad_IOU_good += FP_class_bad_IOU_good
                    all_FP_class_good_IOU_bad += FP_class_good_IOU_bad
                    all_FP_class_IOU_good += FP_class_IOU_good
                    all_FN_class_good_IOU_bad += FN_class_good_IOU_bad
                    all_FN_class_bad_IOU_good += FN_class_bad_IOU_good
                    all_FN_class_bad_IOU_bad += FN_class_bad_IOU_bad

                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() # * inputs.size(0) is removed because YOLOLoss already sums over the batch

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print(f"{phase} Correct Predictions: {all_correct}")
            print(f"{phase} Localization Errors: {all_localization}")
            print(f"{phase} Other Errors: {all_other}")
            print(f"{phase} Other Errors with prob > 0.5: {all_otherbighalf}")
            print(f"{phase} Background Predictions: {all_background}")
            print(f"{phase} Total Bounding Boxes: {all_bounding_boxes}")
            print(f"{phase} TP_class_loca_good: {all_TP_class_loca_good}")
            print(f"{phase} FP_class_bad_IOU_good: {all_FP_class_bad_IOU_good}")
            print(f"{phase} FP_class_good_IOU_bad: {all_FP_class_good_IOU_bad}")
            print(f"{phase} FP_class_IOU_good: {all_FP_class_IOU_good}")
            print(f"{phase} FN_class_good_IOU_bad: {all_FN_class_good_IOU_bad}")
            print(f"{phase} FN_class_bad_IOU_good: {all_FN_class_bad_IOU_good}")
            print(f"{phase} FN_class_bad_IOU_bad: {all_FN_class_bad_IOU_bad}")

            # Save the metrics for this epoch
            losses[phase].append(epoch_loss)
            metrics[phase][epoch] = {
                "correct": all_correct,
                "localization": all_localization,
                "other": all_other,
                "background": all_background,
                "TP_class_loca_good": all_TP_class_loca_good,
                "FP_class_bad_IOU_good": all_FP_class_bad_IOU_good,
                "FP_class_good_IOU_bad": all_FP_class_good_IOU_bad,
                "FP_class_IOU_good": all_FP_class_IOU_good,
                "FN_class_good_IOU_bad": all_FN_class_good_IOU_bad,
                "FN_class_bad_IOU_good": all_FN_class_bad_IOU_good,
            }

            epoch_acc = all_correct / dataset_sizes[phase]

            # Deep copy the model
            if phase == TRAIN and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # write the most recent losses after each epoch
            csvEpochSave = 1
            if epoch % csvEpochSave == 0 and phase == VALIDATION:

                writer.writerow([epoch, 0, epoch_loss,
                                 all_correct, all_localization, all_other,
                                 all_background, all_bounding_boxes, all_TP_class_loca_good, all_FP_class_bad_IOU_good, all_FP_class_good_IOU_bad, all_FP_class_IOU_good, all_FN_class_good_IOU_bad, all_FN_class_bad_IOU_good, all_FN_class_bad_IOU_bad])
            if epoch % csvEpochSave == 0 and phase == TRAIN:

                writer.writerow([epoch, epoch_loss, 0,
                                 all_correct, all_localization, all_other,
                                 all_background, all_bounding_boxes, all_TP_class_loca_good, all_FP_class_bad_IOU_good, all_FP_class_good_IOU_bad, all_FP_class_IOU_good, all_FN_class_good_IOU_bad, all_FN_class_bad_IOU_good, all_FN_class_bad_IOU_bad])

    csv_file.close()

    # Saving the best model
    save_model(best_model_wts, num_epochs=num_epochs, optimizer=optimizer,
               scheduler=scheduler, subset_size=SUBSET_SIZE, batch_size=BATCH_SIZE)

    # Plotting the training results
    plot_training_results(losses, metrics, num_epochs)


def main():

    print("DEVICE:", DEVICE)

    model = Yolo_network()
    model = model.to(DEVICE)  # Use GPU if available

    criterion = YOLOLoss()  # Loss function
    train_images_to_include = ["000000348140.jpg",
                               "000000077332.jpg",
                               "000000058831.jpg",
                               "000000500543.jpg",
                               "000000534275.jpg",
                               "000000470832.jpg",
                               "000000100638.jpg",
                               "000000377183.jpg",
                               "000000278265.jpg",
                               "000000377592.jpg",
                               "000000260223.jpg",
                               "000000050576.jpg",
                               "000000487904.jpg",
                               "000000298933.jpg",
                               "000000336464.jpg",
                               "000000025041.jpg"]


    optimizer = optim.Adam(model.parameters())

    # Start training process
    model = train(
        model, criterion, optimizer, 
        num_epochs=60, fine_tuning_epochs=20,
        # chosen_train_images=train_images_to_include
    )



if __name__ == '__main__':
    main()
