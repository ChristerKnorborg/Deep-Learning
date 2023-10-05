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

        resnet_layers = models.resnet50(pretrained=True)
        # Remove the last layer to get encoder only
        encoder = torch.nn.Sequential(*(list(resnet_layers.children())[:-1]))
        self.encoder = encoder.eval()  # Disable training as encoder is already trained

        # Print the encoder architecture
        ''' print("Encoder: \n")
        print(self.encoder)'''

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

    def forward(self, x):
        # Run data through encoder
        x = self.encoder(x)

        # Run data through decoder
        output = self.fc(x)

        print("Output shape:", output.shape)
        print("Output values:", output)

        return output
