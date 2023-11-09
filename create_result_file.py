import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType, VALIDATION
from model_constants import S, B, DEVICE
from yolo_v1 import Yolo_v1
import json



def run_examples_and_create_file(model_path):


    image_datasets = {
        VALIDATION: DataSetCoco(DataSetType.VALIDATION, training=False)
    }

    dataloaders = {VALIDATION: DataLoader(image_datasets[VALIDATION], shuffle=True)}

    dataset_sizes = {len(image_datasets[VALIDATION])}


    # Put the model in eval mode and perform a forward pass to get the predictions
    model = Yolo_v1()  # create a new instance of your model class
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)  # move model to the intended device
    model.eval()

     # Freeze the pretained encoder weights
    for param in model.encoder.parameters():
            param.requires_grad = False

    # Freeze the fully connected layer weights
    for param in model.fc.parameters():
        param.requires_grad = False

    for _, (inputs, labels) in enumerate(dataloaders[VALIDATION]):
        outputs = model(inputs)
        predictions = torch.squeeze(outputs, 0)

        for i in range(S):
            for j in range(S):
                # Extract data for a single cell
                cell_data = predictions[i, j]

                #JSON data must match COCO format:
                # https://cocodataset.org/#format-results

                # category_id is always 0 as we only guess persons
                data_box1 = [{
                "image_id": int,
                "category_id": 0,
                "bbox": cell_data[1:5],
                "score": cell_data[0],
                }]

                data_box2 = [{
                "image_id": int,
                "category_id": 0,
                "bbox": cell_data[6:10],
                "score": cell_data[5],
                }]

                with open('detections_val2017_YoloV1PersonAUHoldet.json', 'w') as jsonfile:
                    json.dump(data_box1, jsonfile)
                    json.dump(data_box2, jsonfile)


run_examples_and_create_file("./models/model_11-04_08_epoch-60_LR-0.0001_step-none_gamma-none_subset-10000_batch-64.pth")