import torch
import torch.nn as nn



# Loss function based on the loss function from the YOLO paper: https://arxiv.org/pdf/1506.02640.pdf

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YOLOLoss, self).__init__()
        self.S = S # grid size (SxS)
        self.B = B # number of bounding boxes per grid cell
        self.C = C # number of classes

        # lambda_coord and lamda_noobj are parameters used in the paper
        self.lambda_coord = 5 
        self.lambda_noobj = 0.5 

    def forward(self, predictions, target):

        # Split the tensor into its component parts
        # Predictions are in the shape (batch_size, S*S*(B*5 + C))
        pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        pred_class = predictions[..., self.B*5:]

        target_boxes = target[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        target_class = target[..., self.B*5:]

        # Get the objectness score
        obj_score = pred_boxes[..., 4]
        target_obj_score = target_boxes[..., 4]

        # Find the responsible bounding box
        obj_mask = target_obj_score > 0
        noobj_mask = target_obj_score == 0

        # Localization loss
        loc_loss = self.lambda_coord * torch.sum(obj_mask * (torch.sum((pred_boxes[..., :4] - target_boxes[..., :4])**2, dim=-1)))

        # Confidence loss
        conf_loss_obj = torch.sum(obj_mask * (obj_score - target_obj_score)**2)
        conf_loss_noobj = self.lambda_noobj * torch.sum(noobj_mask * (obj_score - target_obj_score)**2)
        conf_loss = conf_loss_obj + conf_loss_noobj

        # Classification loss
        class_loss = torch.sum(obj_mask.view(-1, self.S, self.S, 1).float() * (pred_class - target_class)**2)

        total_loss = loc_loss + conf_loss + class_loss
        return total_loss

# Example usage:
# Assuming predictions and target are your network's output and the ground truth respectively
# predictions = torch.randn((batch_size, 7*7*(2*5 + 20)))
# target = torch.randn((batch_size, 7*7*(2*5 + 20)))
# criterion = YOLOLoss()
# loss = criterion(predictions, target)
