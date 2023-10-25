import torch


S = 7 # grid size (SxS)
B = 2 # number of bounding boxes per grid cell


DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available