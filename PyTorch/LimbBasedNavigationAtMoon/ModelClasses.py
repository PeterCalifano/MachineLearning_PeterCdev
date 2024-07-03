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

# For initialization
import torch.nn.init as init

# %% Custom training function for Moon limb pixel extraction enhancer V3maxDeeper (with target average radius in pixels) - 30-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, deeper fully connected layers, dropout, tanh and leaky ReLU activation
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv3maxDeeper(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, useBatchNorm = False, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        self.useBatchNorm = useBatchNorm

        if useBatchNorm:
            print('Batch normalization layers: enabled.')

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 
        self.batchNormL2 = nn.BatchNorm2d(self.outChannelsSizes[1], eps=1E-5, momentum=0.1, affine=True)

        # Fully Connected predictor
        self.FlattenL3 = nn.Flatten()

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=False)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[3], self.outChannelsSizes[4], bias=True)
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        # Output layer
        #self.batchNormL7 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.DenseOutput = nn.Linear(self.outChannelsSizes[4], 2, bias=True)

    
    def __initialize_weights(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
        # Leaky_ReLU activation layers
        init.kaiming_uniform_(self.conv2dL1, nonlinearity='leaky_relu') 
        init.kaiming_uniform_(self.conv2dL2, nonlinearity='leaky_relu') 
        init.kaiming_uniform_(self.DenseL6, nonlinearity='leaky_relu') 

        # Tanh activation layers
        init.xavier_uniform_(self.DenseL4) 
        init.xavier_uniform_(self.DenseL5) 

    def forward(self, inputSample):
        '''Forward prediction method'''
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))

        # L2
        val = self.conv2dL2(val)
        if self.useBatchNorm:
            val = self.batchNormL2(val)
        val = self.maxPoolL2(torchFunc.leaky_relu(val, self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.DenseL4(val)
        val = self.dropoutL4(val)

        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.tanh(val)

        # L5
        val = self.DenseL5(val)
        val = self.dropoutL5(val)

        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.tanh(val)

        # L6
        val = self.DenseL6(val)
        val = self.dropoutL6(val)

        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = torchFunc.leaky_relu(val, self.alphaLeaky)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer FullyConnected NNv4 (with target average radius in pixels) - 30-06-2024
class HorizonExtractionEnhancer_FullyConnNNv4(nn.Module):
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
    
# %% Custom training function for Moon limb pixel extraction enhancer V3maxDeeper (with target average radius in pixels) - 30-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, deeper fully connected layers, dropout, tanh and leaky ReLU activation and batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv3maxDeeper(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, useBatchNorm = False, poolingKernelSize=2, alphaDropCoeff=0.3, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here?
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=False)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True)

        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[3], self.outChannelsSizes[4], bias=True)
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[4], 2, bias=True)

    def forward(self, inputSample):
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = self.DenseL4(val)
        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.tanh(val)

        # L5
        val = self.dropoutL5(val)
        val = self.DenseL5(val)
        if self.useBatchNorm:
            val = self.batchNormL5(val)

        val = torchFunc.tanh(val)

        # L6 
        val = self.dropoutL6(val)
        val = self.DenseL6(val)
        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = torchFunc.leaky_relu(val, self.alphaLeaky)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection



# %% Custom training function for Moon limb pixel extraction enhancer ResNet-like V5maxDeeper (with target average radius in pixels) - 01-07-2024
class HorizonExtractionEnhancerResCNNv5maxDeeper(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, useBatchNorm = False, poolingKernelSize=2, alphaDropCoeff=0.3, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 6 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here?
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=False)

        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[3], self.outChannelsSizes[4], bias=True)

        # Output layer
        self.batchNormL7 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.DensePreOutput = nn.Linear(self.outChannelsSizes[4]+2, 2, bias=True)  # PATCH CENTRE SKIP CONNECTION
        self.DenseOutput = nn.Linear(2, 2, bias=True)

    def forward(self, inputSample):
        '''Forward prediction method'''
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        device = img2Dinput.device
        contextualInfoInput = torch.tensor([inputSample[:, self.imagePixSize:12], inputSample[:, 14]], dtype=torch.float32, device=device)
        pixPatchCentre = inputSample[:, 12:14]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.tanh(self.DenseL4(val))

        # L5
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = self.dropoutL5(val)
        val = torchFunc.tanh(self.DenseL5(val))

        # L6
        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = self.dropoutL6(val)
        val = torchFunc.leaky_relu(self.DenseL6(val), self.alphaLeaky)

        # Output layer
        if self.useBatchNorm:
            val = self.batchNormL7(val)
    
        # Add pixel Patch centre coordinates
        val = self.DensePreOutput(val)
        val = val + pixPatchCentre
        correctedPix = self.DenseOutput(val)

        return correctedPix
    
class HorizonExtractionEnhancer_ShortCNNv6maxDeeper(nn.Module):
    '''Experimental model with semi-automatic initialization. Structure of the architecture is fixed, but hyperparameters of this are specified at instantiation using
    the "parametersConfig" dictionary with the following keys: kernelSizes, useBatchNorm, poolingKernelSize, alphaDropCoeff, alphaLeaky, patchSize, LinearInputSkipSize'''
    def __init__(self, outChannelsSizes:list, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values
        if 'kernelSizes' not in parametersConfig:
            kernelSizes = [3, 3]
        else:
            kernelSizes = parametersConfig['kernelSizes']

        if 'useBatchNorm' not in parametersConfig:
            useBatchNorm = False
        else:
            useBatchNorm = parametersConfig['useBatchNorm']

        if 'poolingKernelSize' not in parametersConfig:
            poolingKernelSize = 2
        else:
            poolingKernelSize = parametersConfig['poolingKernelSize']

        if 'alphaDropCoeff' not in parametersConfig:
            alphaDropCoeff = 0.1
        else:
            alphaDropCoeff = parametersConfig['alphaDropCoeff']

        if 'alphaLeaky' not in parametersConfig:
            alphaLeaky = 0.01
        else:
            alphaLeaky = parametersConfig['alphaLeaky']
            
        if 'patchSize' not in parametersConfig:
            patchSize = 7
        else:
            patchSize = parametersConfig['patchSize']
            
        assert(len(outChannelsSizes) == 4) # 7 layers in the model
        assert(len(kernelSizes) == 1)
        
        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 1
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[self.numOfConvLayers-1] 
    
        self.LinearInputSkipSize = parametersConfig['LinearInputSkipSize'] #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        idLayer = 0
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)
        idLayer += 1

        #self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        #self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here?
        self.FlattenL2 = nn.Flatten()
        
        self.dropoutL3 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL3 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[idLayer], bias=False)
        self.batchNormL3 = nn.BatchNorm1d(self.outChannelsSizes[idLayer+1], eps=1E-5, momentum=0.1, affine=True)
        idLayer += 1

        self.dropoutL4 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(self.outChannelsSizes[idLayer], self.outChannelsSizes[idLayer+1], bias=True)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[idLayer+1], eps=1E-5, momentum=0.1, affine=True)
        idLayer += 1

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[idLayer], self.outChannelsSizes[idLayer+1], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[idLayer+1], eps=1E-5, momentum=0.1, affine=True)
        idLayer += 1

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[idLayer], 2, bias=True)

    def forward(self, inputSample):
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))

        # Fully Connected Layers
        # L2
        val = self.FlattenL2(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L3 
        #val = self.batchNormL3(val)
        val = self.dropoutL3(val)
        val = self.DenseL3(val)
        if self.useBatchNorm:
            val = self.batchNormL3(val)
        val = torchFunc.tanh(val)

        # L4
        val = self.dropoutL4(val)
        val = self.DenseL4(val)
        if self.useBatchNorm:
            val = self.batchNormL4(val)

        val = torchFunc.tanh(val)

        # L5
        val = self.dropoutL5(val)
        val = self.DenseL5(val)
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.leaky_relu(val, self.alphaLeaky)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection