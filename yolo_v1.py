import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.init as init
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights


from model_constants import S, B, C
class Yolo_v1(nn.Module):


    def __init__(self):
        super(Yolo_v1, self).__init__()

        # Our encoder is a pretrained ResNet-50
        #resnet_layers = models.resnet50(weights=ResNet50_Weights.DEFAULT) # Load pretrained resnet50 model

        resnet_layers = models.resnet18(weights=ResNet18_Weights.DEFAULT) # Use ResNet-18 as the encoder
        self.encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1])) # Remove removes fc layer to get encoder only
        
        # Print the encoder architecture
        #print("Encoder: \n")
        #print(self.encoder)

        # From the paper: S * S * (B * 5 + C), where 5 is the number of parameters in each bounding box (x, y, w, h, confidence)
        prediction_tensor = S**2 * (B*5 + C) # 7*7*(2*5+1) = 539

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            self._make_linear_with_xavier(512, 512), # 2048 is output by resnet encoder. 512 is number of features we want
            #self._make_linear_with_xavier(2048, 512), # 2048 is output by resnet encoder. 512 is number of features we want
            nn.BatchNorm1d(512),
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

        # Reshape the output to [batch_size, S, S, B*5 + C]
        output = output.view(-1, S, S, B*5 + C) # -1 means the dimension is based number of elements in the tensorand  other given dimension 
   
        return output



    def _make_linear_with_xavier(self, in_features, out_features):
        layer = nn.Linear(in_features, out_features)
        init.xavier_uniform_(layer.weight)
        init.constant_(layer.bias, 0)
        return layer


