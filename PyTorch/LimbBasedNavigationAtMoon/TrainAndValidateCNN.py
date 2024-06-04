# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))

import customTorch # Custom torch tools
import limbPixelExtraction_CNN_NN
import datasetPreparation

import torch
import datetime
from torch import nn
from scipy.spatial.transform import Rotation

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

    # Extract data from JSON
    #trainingDataPath = ''
    #validationDataPath = ''
    #trainingDataDict = datasetPreparation.LoadJSONdata(dataPath)
    #validationDataDict = datasetPreparation.LoadJSONdata(dataPath)

    # Construct Dataset objects
    #trainingData    = MoonLimbPixCorrector_Dataset(trainingDataDict)
    #validationData  = MoonLimbPixCorrector_Dataset(validationDataDict)

    ######## TEMPORARY CODE: load single dataset and split ########
    # Create the dataset
    dataPath = os.path.join('/home/peterc/devDir/MATLABcodes/syntheticRenderings/Datapairs')
    dirNamesRoot = os.listdir(dataPath)

    # Select one of the available datapairs folders (each corresponding to a labels generation pipeline output)
    datapairsID = 0
    dataDirPath = os.path.join(dataPath, dirNamesRoot[datapairsID])
    dataFilenames = os.listdir(dataDirPath)

    nImages = len(dataFilenames)

    # DEBUG
    print('Found images:', nImages)
    print(dataFilenames)

    # Get nPatches from the first datapairs files
    dataFileID = 0
    dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
    tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(dataFilePath)

    # DEBUG
    print(tmpdataKeys)

    nPatches = len(tmpdataDict['ui16coarseLimbPixels'][0])
    nSamples = nPatches * nImages

    # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
    # Initialize dataset building variables
    saveID = 0
    
    inputDataArray  = np.zeros((57, nSamples))
    labelsDataArray = np.zeros((6, nSamples))

    for dataPair in dataFilenames:
        # Data dict for ith image
        tmpdataDict = datasetPreparation.LoadJSONdata(dataPath)

        metadataDict = tmpdataDict['metadata']

        dPosCam_TF           = np.array(metadataDict['dPosCam_TF'])
        dAttDCM_fromTFtoCAM  = np.array(metadataDict['dAttDCM_fromTFtoCAM'])
        dSunDir_PixCoords    = np.array(metadataDict['dSunDir_PixCoords'])
        dLimbConic_PixCoords = np.array(metadataDict['dLimbConic_PixCoords'])
        dRmoonDEM = np.array(metadataDict['dRmoonDEM'])

        ui16coarseLimbPixels = np.array(tmpdataDict['ui16coarseLimbPixels'])
        ui8flattenedWindows  = np.array(tmpdataDict['ui8flattenedWindows'])

        for sampleID, patchCentre in enumerate(ui16coarseLimbPixels):
            # Get flattened patch
            flattenedWindow = ui8flattenedWindows[sampleID]

            # Validate patch
            pathIsValid = True # TODO

            if pathIsValid:
                saveID += 1
                inputDataArray[0:48, saveID]  = flattenedWindow
                inputDataArray[49, saveID]    = dRmoonDEM
                inputDataArray[50:51, saveID] = dSunDir_PixCoords
                inputDataArray[52:54, saveID] = Rotation(np.array(dAttDCM_fromTFtoCAM)).as_mrp() # Convert Attitude matrix to MRP parameters
                inputDataArray[55:57, saveID] = dPosCam_TF
                labelsDataArray[:, saveID] = np.flatten(dLimbConic_PixCoords)

    # Shrink dataset remove entries which have not been filled due to invalid path
    print('Number of removed invalid patches:', nSamples - saveID + 1)
    inputDataArray = inputDataArray[:, 0:saveID]           
    labelsDataArray = labelsDataArray[:, 0:saveID]

    ################################################################################################
    return 0

    dataDict['labelsDataArray'] = labelsDataArray 
    dataDict['inputDataArray']  = inputDataArray  
    
    # INITIALIZE DATASET OBJECT
    dataset = customTorch.MoonLimbPixCorrector_Dataset(dataDict)

    # Define the split ratio
    TRAINING_PERC = 0.8

    trainingSize = int(TRAINING_PERC * len(dataset))  
    validationSize = len(dataset) - trainingSize 

    # Split the dataset
    trainingData, validationData = torch.random_split(dataset, [trainingSize, validationSize])

    # Define dataloaders objects
    batch_size = 64 # Defines batch size in dataset
    trainingDataset   = DataLoader(trainingData, batch_size, shuffle=True)
    validationDataset = DataLoader(validationData, batch_size, shuffle=True) 

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION
    # Custom EvalLoss function: MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None)
    lossFcn = customTorch.CustomLossFcn(customTorch.MoonLimbPixConvEnhancer_LossFcn)

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