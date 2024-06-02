# Script created by PeterC 21-05-2024 to experiment with Limb-based navigation using CNN-NN network
# Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225

# BRIEF DESCRIPTION
# The network takes windows of Moon images where the illuminated limb is present, which may be a feature map already identifying the limb if edge detection has been performed.
# Convolutional layers process the image patch to extract features, then stacked with several other inputs downstream. Among these, the relative attitude of the camera wrt to the target body 
# and the Sun direction in the image plane. A series of fully connected layers then map the flatten 1D vector of features plus the contextual information to infer a correction of the edge pixels.
# This correction is intended to remove the effect of the imaged body morphology (i.e., the network will account for the DEM "backward") such that the refined edge pixels better adhere to the
# assumption of ellipsoidal/spherical body without terrain effects. The extracted pixels are then used by the Christian Robinson algorithm to perform derivation of the position vector.

# To insert multiple inputs at different layers of the network, following the discussion:
# 1) Construct the input tensor X such that it can be sliced into the image and the other data
# 2) Define the architecture class
# 3) In defining the forward(self, x) function, slice the x input according to the desired process, reshape and do other operations. 
# 4) Concatenate the removed data with the new inputs from the preceding layers as input to where they need to go.

# Import modules
import torch
import customTorch
import datetime
from torch import nn

from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class


# Example code:
# Note that X given as input to the model is just one, but managed internaly to the class, thus splitting the input as appropriate and only used in the desired layers.

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 20)
        
    def forward(self, x):
        # Use first part of x
        x1 = F.relu(self.fc1(x[:, :8])) # First part of input vector X used as input the the first layer
        x = torch.cat((x1, x[:, 8:]), dim=1)  # Concatenate the result of the first part with the second part of x, which is not processed by the former layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()    
x = torch.randn(1, 12)
output = model(x)