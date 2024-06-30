'''
Script created by PeterC 30-06-2024 to group all model classes for project: Limb-based navigation using CNN-NN network
Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225

# BRIEF DESCRIPTION
The network takes windows of Moon images where the illuminated limb is present, which may be a feature map already identifying the limb if edge detection has been performed.
Convolutional layers process the image patch to extract features, then stacked with several other inputs downstream. Among these, the relative attitude of the camera wrt to the target body 
and the Sun direction in the image plane. A series of fully connected layers then map the flatten 1D vector of features plus the contextual information to infer a correction of the edge pixels.
This correction is intended to remove the effect of the imaged body morphology (i.e., the network will account for the DEM "backward") such that the refined edge pixels better adhere to the
assumption of ellipsoidal/spherical body without terrain effects. The extracted pixels are then used by the Christian Robinson algorithm to perform derivation of the position vector.
Variants using fully connected layers only, without 2D patch information are included as well.
'''

import torch, sys, os
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
from limbPixelExtraction_CNN_NN import *

import datetime
from torch import nn
from math import sqrt
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

from typing import Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim
import torch.nn.functional as torchFunc # Module to apply activation functions in forward pass instead of defining them in the model class


# %% Custom training function for Moon limb pixel extraction enhancer V2max (with target average radius in pixels) - 23-06-2024
class HorizonExtractionEnhancer_FullyConnNNv1(nn.Module):
    def __init__(self, outChannelsSizes:list, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2

        self.LinearInputSize = 8
        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Fully Connected predictor
        self.DenseL1 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[0], bias=False)

        self.dropoutL2 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL2 = nn.Linear(self.outChannelsSizes[0], self.outChannelsSizes[1], bias=True)
        
        self.dropoutL3 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL3 = nn.Linear(self.outChannelsSizes[1], self.outChannelsSizes[2], bias=True)

        self.DenseL4 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        
        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=False)

    def forward(self, inputSample):
        # Get inputs that are not image pixels from input samples
        contextualInfoInput = inputSample[:, self.imagePixSize:] 

        # Fully Connected Layers
        # L1 (Input layer)
        val = torchFunc.tanh(self.DenseL1(contextualInfoInput)) 
        # L2
        val = self.dropoutL2(val)
        val = torchFunc.tanh(self.DenseL2(val))
        # L3
        val = self.dropoutL3(val)
        val = torchFunc.tanh(self.DenseL3(val))
        # L4
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection