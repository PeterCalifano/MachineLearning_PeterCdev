# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import customTorch # Custom torch tools
import limbPixelExtraction_CNN_NN

import torch
import datetime
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim

def main():

    # SETTINGS and PARAMETERS 
    outChannelsSizes = []
    learnRate = 1E-3
    momentumValue = 0.001

    optimizerID = 0

    options = {'taskType': 'classification', 
               'device': customTorch.GetDevice(), 
               'epochs': 10, 
               'Tensorboard':True,
               'saveCheckpoints':True,
               'checkpointsDir:': './checkpoints',
               'modelName': 'trainedModel',
               'loadCheckpoint': False}

    # DATASET LOADING
    # TODO: add datasets
    #trainingData = datasets.FashionMNIST(
    #    root="data",
    #    train=True,
    #    download=True,
    #    transform=ToTensor(),
    #) 
    #testData = datasets.FashionMNIST(
    #    root="data",
    #    train=False,
    #    download=True,
    #    transform=ToTensor(),
    #) 

    trainingData = Dataset()
    testData     = Dataset()

    batch_size = 64 # Defines batch size in dataset

    trainingDataset   = DataLoader(trainingData, batch_size, shuffle=True)
    validationDataset = DataLoader(testData, batch_size, shuffle=True) 

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION
    #loss_fn = nn.CrossEntropyLoss() 
    lossFcn = 0

    # Define optimizer object specifying model instance parameters and optimizer parameters
    if optimizerID == 0:
        optimizer = torch.optim.SGD(modelCNN_NN.parameters(), lr=learnRate, momentum=momentumValue) 
    elif optimizerID == 1:
        optimizer = torch.optim.Adam(modelCNN_NN.parameters(), lr=learnRate)

    # MODEL DEFINITION
    modelCNN_NN = customTorch.HorizonExtractionEnhancerCNN(outChannelsSizes)

    # TRAIN and VALIDATE MODEL
    '''
    TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                              'device': GetDevice(), 
                                                                                                              'epochs': 10, 
                                                                                                              'Tensorboard':True,
                                                                                                              'saveCheckpoints':True,
                                                                                                              'checkpointsDir:': './checkpoints',
                                                                                                              'modelName': 'trainedModel',
                                                                                                              'loadCheckpoint': False}):
    '''
    (trainedModel, trainingLosses, validationLosses) = customTorch.TrainAndValidateModel(dataloaderIndex, modelCNN_NN, lossFcn, optimizer, options)

if __name__ == '__main__':
    main()