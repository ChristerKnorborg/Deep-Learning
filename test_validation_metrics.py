import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType, VALIDATION
from model_constants import S, B, DEVICE
from yolo_v1 import Yolo_v1
import csv
from train import compute_accuracy



def run_examples_and_create_file(model_path):

    image_datasets = {
        VALIDATION: DataSetCoco(DataSetType.VALIDATION, training=False)
    }

    dataset = DataLoader(image_datasets[VALIDATION], shuffle=False, batch_size=64)

    dataloaders = {VALIDATION: dataset} 

    dataset_sizes = {len(image_datasets[VALIDATION])}


    # Put the model in eval mode and perform a forward pass to get the predictions
    model = Yolo_v1()  # create a new instance of your model class
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.to(DEVICE)  # move model to the intended device
    model.eval()

     # Freeze the pretained encoder weights
    for param in model.encoder.parameters():
            param.requires_grad = False

    # Freeze the fully connected layer weights
    for param in model.fc.parameters():
        param.requires_grad = False




     
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

     # Open the CSV file once before you start the epochs.
    csv_file = open("training_metrics_soren.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss',
                     'Correct Predictions', 'Localization Errors',
                     'Other Errors',
                     'Background Predictions', 'Total Bounding Boxes', 'TP_class_loca_good', 'FP_class_bad_IOU_good', 'FP_class_good_IOU_bad', 'FP_class_IOU_good', 'FN_class_good_IOU_bad', 'FN_class_bad_IOU_good', 'FN_class_bad_IOU_bad'])
    
    total_batches = len(dataloaders[VALIDATION])

    for iteration, (inputs, labels, img_id) in enumerate(dataloaders[VALIDATION]):
        print(f"\rProcessing batch {iteration}/{total_batches}", end="")
        outputs = model(inputs)

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


        # write the most recent losses after each epoch
        csvEpochSave = 1

    writer.writerow([0, 0, 0,
                    all_correct, all_localization, all_other,
                    all_background, all_bounding_boxes, all_TP_class_loca_good, 
                    all_FP_class_bad_IOU_good, all_FP_class_good_IOU_bad, all_FP_class_IOU_good,
                        all_FN_class_good_IOU_bad, all_FN_class_bad_IOU_good, all_FN_class_bad_IOU_bad])
    csv_file.close()





run_examples_and_create_file("models/model_11-04_08_epoch-60_LR-0.0001_step-none_gamma-none_subset-10000_batch-64.pth")