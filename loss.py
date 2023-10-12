import torch
import torch.nn as nn


from model_constants import S, B, C

# Loss function based on the loss function from the YOLO paper: https://arxiv.org/pdf/1506.02640.pdf

class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

        # lambda_coord and lamda_noobj are parameters used in the paper for 
        self.lambda_coord = 5 
        self.lambda_noobj = 0.5 
        self.mse = nn.MSELoss(reduction="sum")  # Use sum to sum the losses of all grid cells

    def forward(self, predictions: torch.Tensor, target: torch.Tensor):

        
        # Tensor have shape (batch_size, S, S, C+5*B) where C+5*B is from the formula (C_1,...,C_n, P_c, x, y, w, h), where C_i is the probability of the object being in class i
        # P_c is the probability of the object being present in bouding box B, where every bounding box has 5 values (P_c, x, y, w, h). x and y are the coordinates of the center of the object
        # relative to the grid cell, w and h are the width and height of the bounding box relative to the whole image.


        # Bounding box coordinates
        pred_boxes = predictions[..., C:C+5*B].reshape(-1, S, S, B, 5) # index from C to C+5*B (C inclusive, C+5*B exclusive). E.g. bounding box coordinates
        target_boxes = target[..., C:C+5].reshape(-1, S, S, 1, 5) # index from C to C+5 (C inclusive, C+5 exclusive). E.g. get bounding box for grid cell
        


        # Separate the objectness scores for each bounding box. E.g. make a tensor of shape (batch_size, S, S, 1) where each entry 1 is the probability of the object being present in bouding box B
        pred_confidence1 = predictions[..., 1:2] # Extract index 1 from the last dimension of the tensor (P_c from bounding box 1)
        pred_confidence2 = predictions[..., 6:7] # Extract index 6 from the last dimension of the tensor (P_c from bounding box 2)
        target_confidence = target[..., 1:2] # Extract index 1 (P_c) from the last dimension of the tensor-
    
        
        print("pred_confidence1.shape", pred_confidence1.shape)
        print("pred_confidence1", pred_confidence1)
        print("pred_confidence2.shape", pred_confidence2.shape)
        print("pred_confidence2", pred_confidence2)
        print("target_confidence1.shape", target_confidence.shape)
        print("target_confidence1", target_confidence)

        
        # Separate the components of the target tensor

        # ======================= #
        #   BOX COORDINATE LOSS   #
        # ======================= #

        print("pred_boxes.shape", pred_boxes.shape)
        print("pred_boxes", pred_boxes)
        print("target_boxes.shape", target_boxes.shape)
        print("target_boxes", target_boxes)

        # Determine which bounding box is responsible for the prediction
        pred_bboxes1 = pred_boxes[..., 0, 1:5] # Extract the coordinates from the first bounding boxes
        pred_bboxes2 = pred_boxes[..., 1, 1:5] # Extract the coordinates from the second bounding boxes
        target_bboxes = target_boxes[..., 0, 1:5] # Extract the bounding boxes coordinates from the label
    

        print("pred_bboxes1.shape", pred_bboxes1.shape)
        print("pred_bboxes1", pred_bboxes1)
        print("target_bboxes.shape", target_bboxes.shape)
        print("target_bboxes", target_bboxes)

        # Calculate the IoU for each bounding box in all grid cells
        IoU1 = IoU(pred_bboxes1, target_bboxes) 
        IoU2 = IoU(pred_bboxes2, target_bboxes)

        print("iou1.shape", IoU1.shape)
        print("iou1", IoU1)
        print("iou2.shape", IoU2.shape)
        print("iou2", IoU2)

         # Make a mask of shape (batch_size, S, S, 1) where each entry is True if the first bounding box is responsible for the prediction,
         # and False if the second bounding box is responsible for the prediction
        responsible_box_mask = IoU1 > IoU2

        print("responsible_box_mask.shape", responsible_box_mask.shape)
        print("responsible_box_mask", responsible_box_mask)
        
        # Make a tensor of shape (batch_size, S, S, 1) where each entry is the coordinates of the bounding box that is responsible for the prediction.
        # E.g. some entries will be the coordinates of the first bounding box, and some entries will be the coordinates of the second bounding box
        responsible_pred_bboxes = torch.where(responsible_box_mask[..., None], pred_bboxes1, pred_bboxes2) # 

        print("responsible_pred_bboxes.shape", responsible_pred_bboxes.shape)
        print("responsible_pred_bboxes", responsible_pred_bboxes)
        print("target_bboxes.shape", target_bboxes.shape)
        print("target_bboxes", target_bboxes)
        print("target_boxes.shape", target_boxes.shape)
        print("target_boxes", target_boxes)


        # Compute coordinate loss
        include_tensor = target_confidence > 0 # Object must also be present in the grid cell for the target label to be valid for loss
        include_coords = torch.nonzero(include_tensor.squeeze(-1))
        print("include_coords.shape", include_coords.shape)
        print("include_coords", include_coords)

        # Generate batch indices for each coordinate in include_coords
        batch_size, _, _, _ = responsible_pred_bboxes.shape
        batch_indices = torch.arange(batch_size).view(-1, 1).repeat(1, len(include_coords)).view(-1)

        # Repeat the include_coords for each batch
        repeated_coords = include_coords.repeat(batch_size, 1)

        # Extract the values
        responsible_pred_coords = responsible_pred_bboxes[batch_indices, repeated_coords[:, 0], repeated_coords[:, 1]]
        responsible_target_coords = target_bboxes[batch_indices, repeated_coords[:, 0], repeated_coords[:, 1]]

        
        print("responsible_pred_coords.shape", responsible_pred_coords.shape)
        print("responsible_pred_coords", responsible_pred_coords)
        print("responsible_target_coords.shape", responsible_target_coords.shape)
        print("responsible_target_coords", responsible_target_coords)




        # Separate x, y, w, h
        pred_x, pred_y, pred_w, pred_h = torch.split(responsible_pred_coords, 1, dim=-1)
        target_x, target_y, target_w, target_h = torch.split(responsible_target_coords, 1, dim=-1)


        print("pred_x.shape", pred_x.shape)
        print("pred_x", pred_x)
        print("pred_y.shape", pred_y.shape)
        print("pred_y", pred_y)
        print("pred_w.shape", pred_w.shape)
        print("pred_w", pred_w)
        print("pred_h.shape", pred_h.shape)
        print("pred_h", pred_h)
        
        # Compute losses
        loss_x = self.mse(pred_x, target_x)
        loss_y = self.mse(pred_y, target_y)
        loss_w = self.mse(torch.sqrt(pred_w), torch.sqrt(target_w))
        loss_h = self.mse(torch.sqrt(pred_h), torch.sqrt(target_h))

        print("loss_x", loss_x)
        print("loss_y", loss_y)
        print("loss_w", loss_w)
        print("loss_h", loss_h)

        # Combine the losses
        coord_loss = loss_x + loss_y + loss_w + loss_h
        total_box_coordinate_loss = self.lambda_coord * coord_loss

        print("total_box_coordinate_loss", total_box_coordinate_loss)





        # ======================= #
        #       OBJECT LOSS       #
        # ======================= #

        #pred_obj_conf = pred_boxes[obj_mask, ..., 1:5]
        #target_obj_conf = target_boxes[obj_mask, ..., 1:5]
        #object_loss = self.mse(pred_obj_conf, target_obj_conf)


        # ======================= #
        #     NO OBJECT LOSS      #
        # ======================= #

        #no_obj_mask = target_confidence == 0
        #pred_no_obj_conf = pred_boxes[no_obj_mask, ..., 1:5]
        #target_no_obj_conf = target_boxes[no_obj_mask, ..., 1:5]
        #no_object_loss = self.lambda_noobj * self.mse(pred_no_obj_conf, target_no_obj_conf)


        # ======================= #
        #       CLASS LOSS        #
        # ======================= #


        # ======================= #
        #       TOTAL LOSS        #
        # ======================= #

        #total_loss = coord_loss + object_loss + no_object_loss

        #return total_loss



def IoU(box1, box2):
    # Convert the (x, y, w, h) box format to (xmin, ymin, xmax, ymax)
    box1 = [box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2,
            box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2]
    box2 = [box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2,
            box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2]
    
    # Calculate the intersection coordinates
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area

    return iou



















prediction = torch.Tensor(([
        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[1.0000, 5.0000, 0.6825, 0.5000, 0.1950, 1.0000, 1.0000, 2.0000, 1.0000, 1.0000, 1.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [1.0000, 5.0000, 0.5000, 0.7434, 1.0000, 0.9305, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],  ]))

label = torch.Tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [1.0000, 3.0000, 0.0223, 0.5000, 0.5778, 1.0000],
         [1.0000, 4.0000, 0.5000, 0.5000, 1.0000, 1.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])


criterium = YOLOLoss()

criterium(prediction, label)