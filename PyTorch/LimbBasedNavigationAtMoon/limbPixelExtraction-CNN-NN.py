# Script created by PeterC 21-05-2024 to experiment with Limb-based navigation using CNN-NN network
# Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225
# Prototype architecture designed and coded, 03-05-2024

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
import torch.nn.functional as torchFunc # Module to apply activation functions in forward pass instead of defining them in the model class


# Possible option: X given as input to the model is just one, but managed internally to the class, thus splitting the input as appropriate and only used in the desired layers.
# Alternatively, forward() method can accept multiple inputs based on the definition.

# DEVNOTE: check which modifications are needed for training in mini-batches
# According to GPT4o, the changes required fro the training in batches are not many. Simply build the dataset accordingly, so that Torch is aware of the split:
# Exameple: the forward() method takes two input vectors: forward(self, main_inputs, additional_inputs)
# main_inputs = torch.randn(1000, 784)  # Main input (e.g., image features)
# additional_inputs = torch.randn(1000, 10)  # Additional input (e.g., metadata)
# labels = torch.randint(0, 10, (1000,))
# dataset = TensorDataset(main_inputs, additional_inputs, labels)

class HorizonExtractionEnhancerCNN(nn.Module):
    def __init__(self, outChannelsSizes, kernelSize=3, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.numOfConvLayers = 2
        self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(kernelSize/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputSkipSize = 8
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSize) 
        self.avgPoolL1 = nn.AvgPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSize) 
        self.avgPoolL2 = nn.AvgPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        self.batchNormL3 = nn.BatchNorm1d(self.LinearInputSize, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(self.LinearInputSize, self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, img2Dinput, contextualInfoInput):
        
        # Extract image from inputSample
        #imgInput = inputSample[0:imagePixSize-1] # First portion of the input vector
        #contextualInfoInput = inputSample[imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.avgPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.avgPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    