import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Yolo_v1(nn.Module):
    
    def __init__(self, input_size):
        super(Yolo_v1, self).__init__() 

        resnet_layers = models.resnet50(pretrained=True)
        encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1])) # Remove the last layer to get encoder only
        self.encoder = encoder.eval() # Disable training as encoder is already trained

        # Print the encoder architecture
        print("Encoder: \n")
        print(self.encoder)

        # Add decoder layers to the network
        self.decoder = nn.Sequential(
            # Assuming output from resnet encoder has a depth of 2048
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # Let's assume we want to output 3 channels for RGB
            nn.Conv2d(128, 3, kernel_size=1)
        )

        # Print the decoder architecture
        print("Decoder: \n")
        print(self.decoder)

        
    def forward(self, x):
        # Run data through encoder
        x = self.encoder(x)

        # Run data through decoder
        output = self.decoder(x)

        return output
    

# Try out your network
input_size = 224
model = Yolo_v1(input_size)

