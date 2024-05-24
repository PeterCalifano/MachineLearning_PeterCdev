# Script created by PeterC 17-05-2024 reproducing model presentend in Pugliatti's PhD thesis (C4, Table 4.1)
# Implemented in PyTorch. Datasets on Zenodo: https://zenodo.org/records/7107409

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime


# %% MODEL ARCHITECTURE (CNN ENCODER HEAD FOR U-NET)
numOfInputChannels  = [1, 16, 32, 64, 128] # Number of input channels (3rd dimension if 1 and 2 are the image height and width)
numOfOutputChannels = [16, 32, 64, 128, 256] # Number of output channels resulting from kernel convolution
kernelSize = 3
poolingSize = 2

linearLayerInputSizes = [256*16, 256]
linearLayerOutputSizes = [256, 128]
outputLayerSize = 7


class Conv2dEncoderForUnet(nn.Module):
    def __init__(self, numOfInputChannels, numOfOutputChannels, kernelSize=3, poolingSize=2) -> None:
        super().__init__()
        self.conv2dL1 = nn.Conv2d(numOfInputChannels[0], numOfOutputChannels[0], kernelSize) 
        self.maxPoolL1 = nn.MaxPool2d(poolingSize, poolingSize) # Note: all MaxPooling are [2,2] since the size of the image halves in Mattia's table

        self.conv2dL2 = nn.Conv2d(numOfInputChannels[1], numOfOutputChannels[1], kernelSize) 
        self.maxPoolL2 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL3 = nn.Conv2d(numOfInputChannels[2], numOfOutputChannels[2], kernelSize) 
        self.maxPoolL3 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL4 = nn.Conv2d(numOfInputChannels[3], numOfOutputChannels[3], kernelSize) 
        self.maxPoolL4 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL5 = nn.Conv2d(numOfInputChannels[4], numOfOutputChannels[4], kernelSize) 
        self.maxPoolL5 = nn.MaxPool2d(poolingSize, poolingSize)

        self.dropoutL6 = nn.Dropout2d(0.2)
        self.FlattenL6 = nn.Flatten()

        self.DenseL7 = nn.Linear()
        self.dropoutL7 = nn.Dropout1d(0.2)

        self.DenseL8 = nn.Linear()


    def forward(self, inputSample):

