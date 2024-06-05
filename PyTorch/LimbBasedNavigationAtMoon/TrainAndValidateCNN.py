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
    outChannelsSizes = [32, 64, 75, 15]
    kernelSizes = [3, 1]
    learnRate = 1E-5
    momentumValue = 0.001

    optimizerID = 1

    options = {'taskType': 'regression', 
               'device': customTorch.GetDevice(), 
               'epochs': 100, 
               'Tensorboard':True,
               'saveCheckpoints':True,
               'checkpointsDir': './checkpoints/HorizonPixCorrector_CNN',
               'modelName': 'trainedModel',
               'loadCheckpoint': False,
               'lossLogName': 'Loss_MoonHorizonExtraction',
               'logDirectory': './tensorboardLog'}

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
    
    inputDataArray  = np.zeros((60, nSamples), dtype=np.float32)
    labelsDataArray = np.zeros((11, nSamples), dtype=np.float32)

    for dataPair in dataFilenames:
        # Data dict for ith image
        tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(os.path.join(dataDirPath, dataPair))

        metadataDict = tmpdataDict['metadata']

        dPosCam_TF           = np.array(metadataDict['dPosCam_TF'], dtype=np.float32)
        dAttDCM_fromTFtoCAM  = np.array(metadataDict['dAttDCM_fromTFtoCAM'], dtype=np.float32)
        dSunDir_PixCoords    = np.array(metadataDict['dSunDir_PixCoords'], dtype=np.float32)
        dLimbConic_PixCoords = np.array(metadataDict['dLimbConic_PixCoords'], dtype=np.float32)
        dRmoonDEM            = np.array(metadataDict['dRmoonDEM'], dtype=np.float32)

        ui16coarseLimbPixels = np.array(tmpdataDict['ui16coarseLimbPixels'], dtype=np.float32)
        ui8flattenedWindows  = np.array(tmpdataDict['ui8flattenedWindows'], dtype=np.float32)

        for sampleID in range(ui16coarseLimbPixels.shape[1]):
            # Get flattened patch
            flattenedWindow = ui8flattenedWindows[:, sampleID]

            # Validate patch
            pathIsValid = True # TODO

            if pathIsValid:
                inputDataArray[0:49, saveID]  = flattenedWindow
                inputDataArray[49, saveID]    = dRmoonDEM
                inputDataArray[50:52, saveID] = dSunDir_PixCoords
                inputDataArray[52:55, saveID] = (Rotation.from_matrix(np.array(dAttDCM_fromTFtoCAM))).as_mrp() # Convert Attitude matrix to MRP parameters
                inputDataArray[55:58, saveID] = dPosCam_TF
                inputDataArray[58::, saveID] = ui16coarseLimbPixels[:, sampleID]

                labelsDataArray[0:9, saveID] = np.ravel(dLimbConic_PixCoords)
                labelsDataArray[9:, saveID] = ui16coarseLimbPixels[:, sampleID]

                saveID += 1


    # Shrink dataset remove entries which have not been filled due to invalid path
    print('Number of removed invalid patches:', nSamples - saveID)
    inputDataArray = inputDataArray[:, 0:saveID]           
    labelsDataArray = labelsDataArray[:, 0:saveID]

    ################################################################################################

    dataDict = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
    
    # INITIALIZE DATASET OBJECT
    dataset = customTorch.MoonLimbPixCorrector_Dataset(dataDict)

    # Define the split ratio
    TRAINING_PERC = 0.8

    trainingSize = int(TRAINING_PERC * len(dataset))  
    validationSize = len(dataset) - trainingSize 

    # Split the dataset
    trainingData, validationData = torch.utils.data.random_split(dataset, [trainingSize, validationSize])

    # Define dataloaders objects
    batch_size = 64 # Defines batch size in dataset
    trainingDataset   = DataLoader(trainingData, batch_size, shuffle=True)
    validationDataset = DataLoader(validationData, batch_size, shuffle=True) 

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION
    # Custom EvalLoss function: MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None)
    lossFcn = customTorch.CustomLossFcn(customTorch.MoonLimbPixConvEnhancer_LossFcn)

    # MODEL DEFINITION
    modelCNN_NN = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNN(outChannelsSizes, kernelSizes)

    # Define optimizer object specifying model instance parameters and optimizer parameters
    if optimizerID == 0:
        optimizer = torch.optim.SGD(modelCNN_NN.parameters(), lr=learnRate, momentum=momentumValue) 
    elif optimizerID == 1:
        optimizer = torch.optim.Adam(modelCNN_NN.parameters(), lr=learnRate)


    # TRAIN and VALIDATE MODEL
    '''
    TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                              'device': GetDevice(), 
                                                                                                              'epochs': 10, 
                                                                                                              'Tensorboard':True,
                                                                                                              'saveCheckpoints':True,
                                                                                                              'checkpointsDir': './checkpoints',
                                                                                                              'modelName': 'trainedModel',
                                                                                                              'loadCheckpoint': False}):
    '''
    (trainedModel, trainingLosses, validationLosses) = customTorch.TrainAndValidateModel(dataloaderIndex, modelCNN_NN, lossFcn, optimizer, options)


if __name__ == '__main__':
    main()