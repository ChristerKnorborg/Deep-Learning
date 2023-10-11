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
        


        # Separate the objectness scores for each bounding box. E.g. make a tensor of shape (batch_size, S, S, B, 1) where each entry 1 is the probability of the object being present in bouding box B
        pred_confidence1 = predictions[..., 1:2] # Extract inde 1 from the last dimension of the tensor (P_c from bounding box 1)
        pred_confidence2 = predictions[..., 6:7] # Extract inde 6 from the last dimension of the tensor (P_c from bounding box 2)

        target_confidence1 = target[..., 1:2] # Extract inde 1 from the last dimension of the tensor (P_c from bounding box 1)
        target_confidence2 = target[..., 6:7] # Extract inde 6 from the last dimension of the tensor (P_c from bounding box 2)
        
        
        # Separate the components of the target tensor

        # ======================= #
        #   BOX COORDINATE LOSS   #
        # ======================= #


        # Determine which bounding box is responsible for the prediction

        
        iou1 = IoU(pred_boxes[..., 0, 1:5], target_boxes[..., 0, 1:5]) 
        iou2 = IoU(pred_boxes[..., 1, 1:5], target_boxes[..., 1, 1:5])

        if iou1 > iou2:
            responsible_box_mask1 

        # Only compute loss for cells which contain objects
        obj_mask = target_confidence > 0 # 1 if object exists in cell, 0 otherwise
        pred_coords = pred_boxes[obj_mask, ..., 0] # get the 4 values of the tensor (x, y, w, h)
        target_coords = target_boxes[obj_mask, ..., 0] # get the 4 values of the tensor (x, y, w, h)
        coord_loss = self.lambda_coord * self.mse(pred_coords, target_coords) # MSE loss


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
    # Implement the IoU calculation between box1 and box2
    pass
    return


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

        [[1.0000, 1.0000, 0.6825, 0.5000, 0.1950, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [1.0000, 1.0000, 0.5000, 0.7434, 1.0000, 0.9305, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
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