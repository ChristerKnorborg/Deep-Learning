import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.init as init


class Yolo_v1(nn.Module):

    # TODO figure out how to use the encoder in the beginning(transfer learning)
    # TODO figure out how to couple together images and labels

    def __init__(self):
        super(Yolo_v1, self).__init__()

        # Our encoder is a pretrained ResNet-50
        resnet_layers = models.resnet50(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1])) # Remove remove fc layer to get encoder only
        
        # Print the encoder architecture
        print("Encoder: \n")
        print(self.encoder)


        # Our YOLO v1 specific parameters
        self.S = 7  # Number of grid cells along one dimension (SxS total cells)
        self.B = 2  # Number of bounding boxes per grid cell
        self.C = 1  # Number of classes

        self.fc = nn.Sequential(
            nn.Flatten(),
            self._make_linear_with_xavier(2048, 512),
            nn.Tanh(),
            self._make_linear_with_xavier(512, 1),
            nn.Sigmoid()
        )

    def _make_linear_with_xavier(self, in_features, out_features):
        layer = nn.Linear(in_features, out_features)
        init.xavier_uniform_(layer.weight)
        return layer
        # Print the decoder architecture
        '''print("Decoder: \n")
        print(self.fc)'''


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

        print("Output shape:", output.shape)
        print("Output values:", output)

        return output







Yolo_v1()