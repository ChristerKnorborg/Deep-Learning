import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Yolo_v1(nn.Module):

    #TODO figure out how to use the encoder in the beginning(transfer learning)
    #TODO figure out how to couple together images and labels
    
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
            nn.Linear(2048, 512), # 2048 is the number of features output by the resnet encoder.
                                  # 512 is the number of features we want
            nn.LeakyReLU(0.1), # 0.1 is used in the paper
            # nn.Dropout(0.5), # Optional: for regularization
            nn.Linear(512, self.S**2 * (self.B*5)), # Output layer. S * S ^(B * 5 + C). 7*7*(2*5) = 490. 
                                                    # Where 5 is the number of parameters in each bounding box (x, y, w, h, confidence)
            nn.Sigmoid() # Sigmoid to constrain the output between 0 and 1
        )

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

        return output
    






Yolo_v1()