import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.init as init
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights



from model_constants import S, B
class Yolo_v1(nn.Module):


    def __init__(self):
        super(Yolo_v1, self).__init__()

        # Our encoder is a pretrained ResNet-50
        #resnet_layers = models.resnet50(weights=ResNet50_Weights.DEFAULT) # Load pretrained resnet50 model

        resnet_layers = models.resnet18(weights=ResNet18_Weights.DEFAULT) # Use ResNet-18 as the encoder

        # resnet_layers = models.mobilenet_v2(pretrained = True) # Use MobileNetV2 as the encoder
        self.encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1])) # Remove removes fc layer to get encoder only
        
        # Print the encoder architecture
        #print("Encoder: \n")
        #print(self.encoder)

        # From the paper: S * S * (B * 5), where 5 is the number of parameters in each bounding box (confidence, x, y, w, h)
        prediction_tensor = S**2 * (B*5) # 7*7*(2*5) = 490

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(512),
        #     self._make_linear_with_xavier(512, 512), # 2048 is output by resnet encoder. 512 is number of features we want
        #     #self._make_linear_with_xavier(2048, 512), # 2048 is output by resnet encoder. 512 is number of features we want
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.1), # 0.1 is used in the paper
        #     # nn.Dropout(0.5), # Optional: for regularization
        #     self._make_linear_with_xavier(512, prediction_tensor),
        #     nn.Sigmoid() # Sigmoid to constrain the output between 0 and 1
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1), # 0.1 is used in the paper
            nn.BatchNorm1d(512),
            nn.Linear(512, prediction_tensor),
            nn.Sigmoid() # Sigmoid to constrain the output between 0 and 1
        )

        self.apply(self._xavier_init) # Initialize the weights of the fully connected layer with Xavier initialization




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
        output = output.view(-1, S, S, B*5) # -1 means the dimension is based number of elements in the tensor and other given dimension 
   
        return output



    def _xavier_init(self, m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize biases to 0

