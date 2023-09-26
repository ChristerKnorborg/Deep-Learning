import torch
import torch.nn as nn
import torchvision.models as models


# 
encoder = models.resnet50(pretrained=True) # Check available models here: https://pytorch.org/vision/0.15/models

# Print the architecture
print(encoder)

encoder = torch.nn.Sequential(*(list(encoder.children())[:-1])) # Remove the last layer to get encoder

# Print the architecture
print(encoder)