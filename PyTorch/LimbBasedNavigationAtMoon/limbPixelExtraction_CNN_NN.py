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
import customTorchTools
import datetime
from torch import nn
from math import sqrt
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

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

# %% Custom Dataset class for Moon Limb pixel extraction CNN enhancer - 01-06-2024
# First prototype completed by PC - 04-06-2024 --> to move to new module
class MoonLimbPixCorrector_Dataset():

    def __init__(self, dataDict:dict, datasetType:str='train', transform=None, target_transform=None):
            # Store input and labels sources
            self.labelsDataArray = dataDict['labelsDataArray']
            self.inputDataArray = dataDict['inputDataArray']

            # Initialize transform objects
            self.transform = transform
            self.target_transform = target_transform

            # Set the dataset type (train, test, validation)
            self.datasetType = datasetType

    def __len__(self):
        return np.shape(self.labelsDataArray)[1]

    # def __getLabelsData__(self):
    #     self.labelsDataArray

    def __getitem__(self, index):
        label   = self.labelsDataArray[:, index]
        inputVec = self.inputDataArray[:, index]

        return inputVec, label


# %% Custom loss function for Moon Limb pixel extraction CNN enhancer - 01-06-2024
def MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    coeff = 0.98 # TODO: convert params to dict
    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    # Evaluate loss
    conicLoss = 0.0 # Weighting violation of Horizon conic equation
    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=customTorchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

    L2regLoss = torch.norm(predictCorrection)**2 # Weighting the norm of the correction to keep it as small as possible
    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss
    return lossValue

# %% Custom loss function for Moon Limb pixel extraction CNN enhancer with strong loss for out-of-patch predictions - 23-06-2024
def MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm(predictCorrection, labelVector, params:list=None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    coeff = 0.98 # TODO: convert params to dict
    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    # Evaluate loss
    conicLoss = 0.0 # Weighting violation of Horizon conic equation
    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=customTorchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

    L2regLoss = torch.norm(predictCorrection)**2 # Weighting the norm of the correction to keep it as small as possible
    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss + outOfPatchoutLoss_RectExp(predictCorrection, 7)
    
    return lossValue

# %% Additional Loss term for out-of-patch predictions based on Rectified Exponential function - 23-06-2024
def outOfPatchoutLoss_RectExp(predictCorrection, patchSize=7, slopeMultiplier=2):
    if predictCorrection.size()[1] != 2:
        raise ValueError('predictCorrection must have 2 columns for x and y pixel correction')
    
    lossValue = 0.0
    halfPatchSize = patchSize/2

    # Compute the out-of-patch loss 
    if predictCorrection[0] > halfPatchSize:
        lossValue += torch.exp(2*(predictCorrection[0] - halfPatchSize))

    if predictCorrection[1] > halfPatchSize:
        lossValue += torch.exp(2*(predictCorrection[1] - halfPatchSize))

    return lossValue/2.0 # Return the average of the two losses

# %% Custom normalized loss function for Moon Limb pixel extraction CNN enhancer - 23-06-2024
def MoonLimbPixConvEnhancer_NormalizedLossFcn(predictCorrection, labelVector, params:list=None):
    
    # TODO

    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    coeff = 0.98 # TODO: convert params to dict
    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    # Evaluate loss
    conicLoss = 0.0 # Weighting violation of Horizon conic equation
    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=customTorchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        #conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

    #L2regLoss = torch.norm(predictCorrection)**2 # Weighting the norm of the correction to keep it as small as possible
    # Total loss function
    #lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss 

    return lossValue

# %% Function to validate path (check it is not completely black or white)
def IsPatchValid(patchFlatten, lowerIntensityThr=3):
    
    # Count how many pixels are below threshold
    howManyBelowThreshold = np.sum(patchFlatten <= lowerIntensityThr)
    howManyPixels = len(patchFlatten)
    width = np.sqrt(howManyPixels)
    lowerThreshold = width/2
    upperThreshold = howManyPixels - lowerThreshold
    if howManyBelowThreshold <  lowerThreshold or howManyBelowThreshold > upperThreshold:
        return False
    else:
        return True

# %% ARCHITECTURES ############################################################################################################

# %% Custom CNN-NN model for Moon limb pixel extraction enhancer - 01-06-2024
class HorizonExtractionEnhancerCNN(nn.Module):
    '''Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
    '''
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = 32 # Need to make this automatic... computing the number of features from the last convolutional layer (flatten of the volume)
        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        firstIndex = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(firstIndex, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]

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
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2 (with target average radius in pixels) - 21-06-2024
'''
Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv2(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = 32 # Need to make this automatic... computing the number of features from the last convolutional layer (flatten of the volume)
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        firstIndex = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(firstIndex, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]

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
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
# %% Custom CNN-NN model for Moon limb pixel extraction enhancer V1max - 23-06-2024
class HorizonExtractionEnhancerCNN(nn.Module):
    '''Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
    '''
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = 32 # Need to make this automatic... computing the number of features from the last convolutional layer (flatten of the volume)
        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        firstIndex = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(firstIndex, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

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
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2max (with target average radius in pixels) - 23-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv2(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = 32 # Need to make this automatic... computing the number of features from the last convolutional layer (flatten of the volume)
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        firstIndex = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(firstIndex, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

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
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom CNN-NN model for Moon limb pixel extraction enhancer - V1.3 Increased CNN encoder size - 24-06-2024
class HorizonExtractionEnhancerCNN(nn.Module):
    '''Architecture characteristics: Increased size of Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
    '''
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = 32 # Need to make this automatic... computing the number of features from the last convolutional layer (flatten of the volume)
        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingSize, poolingSize)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingSize, poolingSize) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        firstIndex = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(firstIndex, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]

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
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
