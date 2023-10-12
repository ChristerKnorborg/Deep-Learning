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
        target_boxes = target[..., C:C+5*B].reshape(-1, S, S, B, 5) # index from C to C+5*B (C inclusive, C+5*B exclusive). E.g. get every bounding box per grid cell
        


        # Separate the objectness scores for each bounding box. E.g. make a tensor of shape (batch_size, S, S, 1) where each entry 1 is the probability of the object being present in bouding box B
        pred_confidence1 = predictions[..., 1:2] # Extract index 1 from the last dimension of the tensor (P_c from bounding box 1)
        pred_confidence2 = predictions[..., 6:7] # Extract index 6 from the last dimension of the tensor (P_c from bounding box 2)

        target_confidence1 = target[..., 1:2] # Extract index 1 from the last dimension of the tensor (P_c from bounding box 1)
        target_confidence2 = target[..., 6:7] # Extract index 6 from the last dimension of the tensor (P_c from bounding box 2)
        
        print("pred_confidence1.shape", pred_confidence1.shape)
        print("pred_confidence1", pred_confidence1)
        print("target_confidence1.shape", target_confidence1.shape)

        
        # Separate the components of the target tensor

        # ======================= #
        #   BOX COORDINATE LOSS   #
        # ======================= #


        # Determine which bounding box is responsible for the prediction
        pred_bboxes1 = pred_boxes[..., 0, 1:5] # Extract the coordinates from the first bounding boxes
        pred_bboxes2 = pred_boxes[..., 1, 1:5] # Extract the coordinates from the second bounding boxes
        target_bboxes1 = target_boxes[..., 0, 1:5] # Extract the coordinates from the first bounding boxes
        target_bboxes2 = target_boxes[..., 1, 1:5] # Extract the coordinates from the second bounding boxes

        print("pred_bboxes1.shape", pred_bboxes1.shape)
        print("pred_bboxes1", pred_bboxes1)

        # Calculate the IoU for each bounding box in all grid cells
        iou1 = IoU(pred_bboxes1, target_bboxes1) 
        iou2 = IoU(pred_bboxes2, target_bboxes2)

        print("iou1.shape", iou1.shape)
        print("iou1", iou1)



        responsible_box_mask1 = iou1 > iou2 # Mask of shape (batch_size, S, S, 1) where each entry is 1 if the first bounding box is responsible for the prediction, 0 otherwise
        responsible_box_mask2 = ~responsible_box_mask1 

        # Compute coordinate loss for bounding box 1
        obj_mask1 = (target_confidence1 > 0) & responsible_box_mask1
        pred_coords1 = pred_boxes[obj_mask1, ..., 0, 1:5]
        target_coords1 = target_boxes[obj_mask1, ..., 0, 1:5]

        # Separate x, y, w, h
        pred_x1, pred_y1, pred_w1, pred_h1 = torch.split(pred_coords1, 1, dim=-1)
        target_x1, target_y1, target_w1, target_h1 = torch.split(target_coords1, 1, dim=-1)

        # Compute losses
        loss_x1 = self.mse(pred_x1, target_x1)
        loss_y1 = self.mse(pred_y1, target_y1)
        loss_w1 = self.mse(torch.sqrt(pred_w1), torch.sqrt(target_w1))
        loss_h1 = self.mse(torch.sqrt(pred_h1), torch.sqrt(target_h1))

        # Combine the losses
        coord_loss1 = loss_x1 + loss_y1 + loss_w1 + loss_h1

        # Compute coordinate loss for bounding box 2
        obj_mask2 = (target_confidence2 > 0) & responsible_box_mask2
        pred_coords2 = pred_boxes[obj_mask2, ..., 1, 1:5]
        target_coords2 = target_boxes[obj_mask2, ..., 1, 1:5]

        # Separate x, y, w, h
        pred_x2, pred_y2, pred_w2, pred_h2 = torch.split(pred_coords2, 1, dim=-1)
        target_x2, target_y2, target_w2, target_h2 = torch.split(target_coords2, 1, dim=-1)

        # Compute losses
        loss_x2 = self.mse(pred_x2, target_x2)
        loss_y2 = self.mse(pred_y2, target_y2)
        loss_w2 = self.mse(torch.sqrt(pred_w2), torch.sqrt(target_w2))
        loss_h2 = self.mse(torch.sqrt(pred_h2), torch.sqrt(target_h2))

        # Combine the losses
        coord_loss2 = loss_x2 + loss_y2 + loss_w2 + loss_h2

        # Sum up the losses and weight with lambda_coord
        total_coord_loss = self.lambda_coord * (coord_loss1 + coord_loss2)




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



















label = torch.Tensor(([
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

        [[1.0000, 5.0000, 0.6825, 0.5000, 0.1950, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
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



criterium = YOLOLoss()

criterium(label, label)