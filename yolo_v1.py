import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.init as init
from torchvision.models.resnet import ResNet18_Weights


from model_constants import S, B


class Yolo_v1(nn.Module):

    def __init__(self):
        super(Yolo_v1, self).__init__()


        # Use ResNet-18 as the encoder
        resnet_layers = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # resnet_layers = models.mobilenet_v2(pretrained = True) # Use MobileNetV2 as the encoder
        # Remove removes fc layer to get encoder only
        self.encoder = torch.nn.Sequential(
            *(list(resnet_layers.children())[:-1]))

        # Print the encoder architecture
        # print("Encoder: \n")
        # print(self.encoder)

        # From the paper: S * S * (B * 5), where 5 is the number of parameters in each bounding box (confidence, x, y, w, h)
        prediction_tensor = S**2 * (B*5)  # 7*7*(2*5) = 490



        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),  # 0.1 is used in the paper
            self.xavier_linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),  # 0.1 is used in the paper
            self.xavier_linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),  # 0.1 is used in the paper
            self.xavier_linear(512, prediction_tensor),
            nn.BatchNorm1d(prediction_tensor),
            # self.xavier_linear(512, prediction_tensor),
            nn.Sigmoid()  # Sigmoid to constrain the output between 0 and 1
        )

        # Freeze the pretained encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the fully connected layer weights for training
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Run data through encoder
        x = self.encoder(x)

        # Run data through decoder
        output = self.fc(x)

        # Reshape the output to [batch_size, S, S, B*5]
        # -1 means the dimension is based number of elements in the tensor and other given dimension
        output = output.view(-1, S, S, B*5)

        return output

    def xavier_linear(self, in_dim, out_dim):
        linear = nn.Linear(in_dim, out_dim)
        init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            init.constant_(linear.bias, 0)
        return linear
