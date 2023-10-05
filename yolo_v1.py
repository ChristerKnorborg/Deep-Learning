import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Yolo_v1(nn.Module):

    #TODO figure out how to use the encoder in the beginning(transfer learning)
    #TODO figure out how to couple together images and labels
    
    def __init__(self):
        super(Yolo_v1, self).__init__() 

        resnet_layers = models.resnet50(pretrained=True)
        encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1])) # Remove the last layer to get encoder only
        self.encoder = encoder.eval() # Disable training as encoder is already trained

        # Print the encoder architecture
        ''' print("Encoder: \n")
        print(self.encoder)'''

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Optional: for regularization
            nn.Linear(512, 1) # 1 is the number of classes
        )

        # Print the decoder architecture
        '''print("Decoder: \n")
        print(self.fc)'''

        
    def forward(self, x):
        # Run data through encoder
        x = self.encoder(x)

        # Run data through decoder
        output = self.fc(x)

        return output
    



