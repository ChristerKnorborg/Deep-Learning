import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DataSetCoco, DataSetType, VALIDATION
from model_constants import S, B, DEVICE
from yolo_v1 import Yolo_v1
import json

def convert_to_coco_format(bbox, cell_index, img_width, img_height, S):
    # bbox contains [center-x, center-y, width, height] relative to the cell
    # cell_index is a tuple (i, j) representing the cell's row and column index
    i, j = cell_index
    cell_size = img_width / S

    # Convert the center-x and center-y to absolute coordinates
    x_center = (bbox[0] + j) * cell_size
    y_center = (bbox[1] + i) * cell_size

    # Convert width and height to absolute dimensions
    w = bbox[2] * img_width
    h = bbox[3] * img_height

    # Convert to top-left coordinates
    x_top_left = x_center - (w / 2)
    y_top_left = y_center - (h / 2)

    return [x_top_left, y_top_left, w, h]



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




     
    # Create an empty list to store all detection boxes
    all_data_boxes = []   

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


                img_width, img_height = inputs.shape[2], inputs.shape[3]  # Assuming inputs is a tensor of shape [batch_size, channels, height, width]
                bbox = convert_to_coco_format(bbox, (i,j), img_width, img_height, S)

                #JSON data must match COCO format:
                # https://cocodataset.org/#format-results




                # category_id is always 0 as we only guess persons
                data_box = {
                "image_id": img_id.item(),
                "category_id": 1,
                "bbox": bbox,
                "score": confidence,
                }

                all_data_boxes.append(data_box)


    with open('detections_val2017_YoloV1PersonAUHoldet.json', 'w') as jsonfile:
                    json.dump(all_data_boxes, jsonfile)      


run_examples_and_create_file("./models/model_11-04_08_epoch-60_LR-0.0001_step-none_gamma-none_subset-10000_batch-64.pth")