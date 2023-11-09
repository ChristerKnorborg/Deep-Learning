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

    dataset = DataLoader(image_datasets[VALIDATION], shuffle=True)

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

    for _, (inputs, labels, img_id) in enumerate(dataloaders[VALIDATION]):
        outputs = model(inputs)
        predictions = torch.squeeze(outputs, 0)
        dataloaders[VALIDATION].dataset


        for i in range(S):
            for j in range(S):
                # Extract data for a single cell
                cell_data = predictions[i, j]


                # Pick the box with highest score
                 # Two sets of predictions; only consider the one with the highest confidence
                if cell_data[0] > cell_data[5]:
                    confidence = cell_data[0].item()
                    bbox = cell_data[1:5].tolist()
                else:
                    confidence = cell_data[5].item()
                    bbox = cell_data[6:10].tolist()

                #JSON data must match COCO format:
                # https://cocodataset.org/#format-results

                # category_id is always 0 as we only guess persons
                data_box = [{
                "image_id": img_id.item(),
                "category_id": 0,
                "bbox": bbox,
                "score": confidence,
                }]

                with open('detections_val2017_YoloV1PersonAUHoldet.json', 'w') as jsonfile:
                    json.dump(data_box, jsonfile)


run_examples_and_create_file("./models/model_11-04_08_epoch-60_LR-0.0001_step-none_gamma-none_subset-10000_batch-64.pth")