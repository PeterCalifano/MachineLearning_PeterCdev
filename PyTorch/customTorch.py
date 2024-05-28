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

import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class

# class FDNNbuilder:
#     def __init__():

# %% Function to get device if not passed to trainModel and validateModel()
def getDevice():
    device = ("cuda:0"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu" )
    print(f"Using {device} device")
    return device

# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
def trainModel(dataloader:DataLoader, model:nn.Module, lossFcn, optimizer, device=getDevice()):
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

# %% Function to validate model using dataset and specified loss function - 04-05-2024
def validateModel(dataloader:DataLoader, model:nn.Module, lossFcn, device=getDevice()):
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
            # Determine if prediction is correct and accumulate
            # Explanation: get largest output logit (the predicted class) and compare to Y. 
            # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
            correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item() 

    testLoss/=numberOfBatches # Compute batch size normalized loss value
    correctOuputs /= size # Compute percentage of correct classifications over batch size
    print(f"Test Error: \n Accuracy: {(100*correctOuputs):>0.1f}%, Avg loss: {testLoss:>8f} \n")

# %% Function to save model state - 04-05-2024
def SaveModelState(model:nn.Module, modelName:str="trainedModel"):
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


def LoadModelState(model:nn.Module, modelName:str="trainedModel", filepath:str="testModels/"):

    # Contatenate file path
    modelPath = filepath + modelName + ".pth"
    # Load model from file
    model.load_state_dict(torch.load(modelPath))
    # Evaluate model to set (weights, biases)
    model.eval()


# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
from abc import abstractmethod
from abc import ABCMeta

class GenericSupervisedDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, inputDataPath:str, labelsDataPath:str, transform=None, target_transform=None):
        # Store input and labels sources
        self.labelsDir = labelsDataPath
        self.inputDir = inputDataPath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    @abstractmethod
    def __getLabelsData__(self):
        raise NotImplementedError()
        # Get and store labels vector
        self.labels # "Read file" of some kind goes here. Best current option: write to JSON

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
        return inputVec, label


class MoonLimbPixCorrector_Dataset(GenericSupervisedDataset):
    
    def __getLabelsData__(self):
        return super().__getLabelsData__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)


# TODO
#def ConfigTensorboardSession():
#def ControlArgsParser():

