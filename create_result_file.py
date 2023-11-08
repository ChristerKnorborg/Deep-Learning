import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType, TEST
from model_constants import S, B, DEVICE
from yolo_v1 import Yolo_v1
import json



def run_test_examples(model_path):


    image_datasets = {
        TEST: DataSetCoco(DataSetType.TEST, training=False)
    }

    dataloaders = {DataLoader(image_datasets[TEST], shuffle=True)}

    dataset_sizes = {len(image_datasets[TEST])}


    # Put the model in eval mode and perform a forward pass to get the predictions
    model = Yolo_v1()  # create a new instance of your model class
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)  # move model to the intended device
    model.eval()

    for _, (inputs, labels) in enumerate(dataloaders[TEST], start=1):
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

                with open('resultfile.txt', 'w') as txtfile:
                    json.dump(data_box1, txtfile)
                    json.dump(data_box2, txtfile)