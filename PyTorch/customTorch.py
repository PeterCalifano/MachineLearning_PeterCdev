# Module collecting utilities function building upon PyTorch to speed up prototyping, training and testing of Neural Nets 
# Created by PeterC - 04-05-2024

# Import modules
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime
import numpy as np
import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class

# class FDNNbuilder:
#     def __init__():

# %% Function to get device if not passed to trainModel and validateModel()
def GetDevice():
    device = ("cuda:0"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu" )
    print(f"Using {device} device")
    return device

# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
# TO REWORK (make it more general)
def TrainModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, optimizer, device=GetDevice()):

    size=len(dataloader.dataset) # Get size of dataloader dataset object

    model.train() # Set model instance in training mode ("informing" backend that the training is going to start)
    for batch, (X, Y) in enumerate(dataloader): # Recall that enumerate gives directly both ID and value in iterable object

        X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device

        # Perform FORWARD PASS
        predVal = model(X) # Evaluate model at input
        loss = lossFcn(predVal, Y) # Evaluate loss function to get loss value (this returns loss function instance, not a value)

        # Perform BACKWARD PASS
        loss.backward() # Compute gradients
        optimizer.step() # Apply gradients from the loss
        optimizer.zero_grad() # Reset gradients for next iteration

        if batch % 100 == 0: # Print loss value every 100 steps
            loss, currentStep = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{currentStep:>5d}/{size:>5d}]")

        # TODO: add command for Tensorboard here

# %% Function to validate model using dataset and specified loss function - 04-05-2024
# TO REWORK (make it more general)
def ValidateModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, device=GetDevice(), taskType:str='classification'):
    size = len(dataloader.dataset) 
    numberOfBatches = len(dataloader)
    model.eval() # Set the model in evaluation mode
    testLoss, correctOuputs = 0, 0 # Accumulation variables

    with torch.no_grad(): # Tell torch that gradients are not required
  
        for X,Y in dataloader:

            X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device
            # Perform FORWARD PASS
            predVal = model(X) # Evaluate model at input
            testLoss += lossFcn(predVal, Y).item() # Evaluate loss function and accumulate

            # TODO: MODIFY BASED ON PROBLEM TYPE
            if taskType.lower() == 'classification': 
                # Determine if prediction is correct and accumulate
                # Explanation: get largest output logit (the predicted class) and compare to Y. 
                # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
                correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item() 

            elif taskType.lower() == 'regression':
                print('TODO')


    if taskType.lower() == 'classification': 
        # TODO: MODIFY BASED ON PROBLEM TYPE
        testLoss/=numberOfBatches # Compute batch size normalized loss value
        correctOuputs /= size # Compute percentage of correct classifications over batch size

    elif taskType.lower() == 'regression':
        print('TODO')

    print(f"Test Error: \n Accuracy: {(100*correctOuputs):>0.1f}%, Avg loss: {testLoss:>8f} \n")
    # TODO: add command for Tensorboard here


# %% Class to define a custom loss function for training, validation and testing - 01-06-2024
# NOTE: Function EvalLossFcn must be implemented using Torch operations to work!

class CustomLossFcn(nn.Module):
    # Class constructor
    def __init__(self, EvalLossFcn:function) -> None:
        super(CustomLossFcn, self).__init__() # Call constructor of nn.Module
        if len(EvalLossFcn) >= 2:
            self.LossFcnObj = EvalLossFcn
        else: 
            raise ValueError('Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')    

    # Forward Pass evaluation method using defined EvalLossFcn
    def forward(self, predictVector, labelVector, params=None):
        lossBatch = self.LossFcnObj(predictVector, labelVector, params)
        return lossBatch.mean()
   
# %% Custom loss function for Moon Limb pixel extraction CNN enhancer - 01-06-2024
def MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None):
    # alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    coeff = 0.5
    LimbConicMatrixImg = np.reshape(labelVector, 3, 3)
    # Evaluate loss
    conicLoss = torch.matmul(predictCorrection.T, torch.matmul(LimbConicMatrixImg, predictCorrection)) # Weighting violation of Horizon conic equation
    L2regLoss = torch.norm(predictCorrection)**2 # Weighting the norm of the correction to keep it as small as possible

    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss
    return lossValue


# %% Function to save model state - 04-05-2024
def SaveModelState(model:nn.Module, modelName:str="trainedModel") -> None:
    import os.path
    if not(os.path.isdir('./testModels')):
        os.mkdir('testModels')
        if not(os.path.isfile('.gitignore')):
            # Write gitignore in the current folder if it does not exist
            gitignoreFile = open('.gitignore', 'w')
            gitignoreFile.write("\ntestModels/*")
            gitignoreFile.close()
        else:
            # Append to gitignore if it exists
            gitignoreFile = open('.gitignore', 'a')
            gitignoreFile.write("\ntestModels/*")
            gitignoreFile.close()
        

    currentTime = datetime.datetime.now()
    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    filename = "testModels/" + modelName + "_" + formattedTimestamp
    print("Saving PyTorch Model State to", filename)
    torch.save(model.state_dict(), filename) # Save model as internal torch representation

# %% Function to load model state - 04-05-2024 
def LoadModelState(model:nn.Module, modelName:str="trainedModel", filepath:str="testModels/") -> nn.Module:
    # Contatenate file path
    modelPath = filepath + modelName + ".pth"
    # Load model from file
    model.load_state_dict(torch.load(modelPath))
    # Evaluate model to set (weights, biases)
    model.eval()
    return model

# %% Function to save Dataset object - 01-06-2024
def SaveTorchDataset(datasetObj:Dataset, datasetFilePath:str) -> None:
    torch.save(datasetObj, datasetFilePath + ".pt")

# %% Function to load Dataset object - 01-06-2024
def LoadTorchDataset(datasetFilePath:str) -> Dataset:
    return torch.load(datasetFilePath + ".pt")


# %% Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
from abc import abstractmethod
from abc import ABCMeta


class GenericSupervisedDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, inputDataPath:str='inputData/', labelsDataPath:str='labelsData/', 
                 datasetType:str='train', transform=None, target_transform=None):
        # Store input and labels sources
        self.labelsDir = labelsDataPath
        self.inputDir = inputDataPath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

        # Set the dataset type (train, test, validation)
        self.datasetType = datasetType

    def __len__(self):
        return len() # TODO

    @abstractmethod
    def __getLabelsData__(self):
        raise NotImplementedError()
        # Get and store labels vector
        self.labels # TODO: "Read file" of some kind goes here. Best current option: write to JSON 

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
        return inputVec, label

# %% Custom Dataset class for Moon Limb pixel extraction CNN enhancer - 01-06-2024
class MoonLimbPixCorrector_Dataset(GenericSupervisedDataset):

    def __getLabelsData__(self):
        return super().__getLabelsData__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)


# TODO
#def ConfigTensorboardSession():
#def ControlArgsParser():

