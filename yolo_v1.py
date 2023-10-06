import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.init as init


class Yolo_v1(nn.Module):


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

        # From the paper: S * S * (B * 5 + C), where 5 is the number of parameters in each bounding box (x, y, w, h, confidence)
        prediction_tensor = self.S**2 * (self.B*5 + self.C) # 7*7*(2*5) = 490.

        self.fc = nn.Sequential(
            nn.Flatten(),
            self._make_linear_with_xavier(2048, 512), # 2048 is output by resnet encoder. 512 is number of features we want
            nn.LeakyReLU(0.1), # 0.1 is used in the paper
            # nn.Dropout(0.5), # Optional: for regularization
            self._make_linear_with_xavier(512, prediction_tensor),
            nn.Sigmoid() # Sigmoid to constrain the output between 0 and 1
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

        print("Output shape:", output.shape)
        print("Output values:", output)

        return output



    def _make_linear_with_xavier(self, in_features, out_features):
        layer = nn.Linear(in_features, out_features)
        init.xavier_uniform_(layer.weight)
        return layer
        # Print the decoder architecture
        '''print("Decoder: \n")
        print(self.fc)'''


