# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import sys, os, multiprocessing
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchToolsTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import customTorchTools # Custom torch tools
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

import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim

def main(id):

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    TRAINING_PERC = 0.80
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [16, 32, 75, 15] 
    kernelSizes = [3, 1]
    learnRate = 1E-10
    momentumValue = 0.001

    optimizerID = 1 # 0

    device = customTorchTools.GetDevice()

    exportTracedModel = True

    # Options to restart training from checkpoint
    if id == 0:
        runID = str(0)
        #modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2_run3'
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2max_largerCNN_run' + runID
        tensorboardLogDir = './tensorboardLog_v2max_largerCNN_run' + runID
        modelArchName = 'HorizonPixCorrector_CNNv2max_largerCNN_run' + runID
        inputSize = 56 # TODO: update this according to new model

        sys.stdout = open("stdout_log_" + modelArchName + '.txt', 'w') # Redirect print outputs


    elif id == 1:
        runID = str(0)
        inputSize = 57 # TODO: update this according to new model
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv3max_largerCNN_run' + runID
        tensorboardLogDir = './tensorboardLog_v3max_largerCNN_run'   + runID
        modelArchName = 'HorizonPixCorrector_CNNv3max_largerCNN_run' + runID

        sys.stdout = open("stdout_log_" + modelArchName + '.txt', 'w') # Redirect print outputs



    options = {'taskType': 'regression', 
               'device': device, 
               'epochs': 5, 
               'Tensorboard':True,
               'saveCheckpoints':True,
               'checkpointsOutDir': modelSavePath,
               'modelName': modelArchName,
               'loadCheckpoint': False,
               'checkpointsInDir': modelSavePath,
               'lossLogName': 'Loss_MoonHorizonExtraction',
               'logDirectory': tensorboardLogDir,
               'epochStart': 0}



    if options['epochStart'] == 0:
        restartTraining = False
    else:
        restartTraining = True
        # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
        with os.scandir(modelSavePath) as it:
            modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]


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
    datasetID = 6 # ACHTUNG! paths from listdir are randomly ordered! --> TODO: modify
    dataDirPath = os.path.join(dataPath, dirNamesRoot[datasetID])

    dataFilenames = os.listdir(dataDirPath)

    nImages = len(dataFilenames)

    # DEBUG
    print('Found images:', nImages)
    print(dataFilenames)

    # Get nPatches from the first datapairs files
    dataFileID = 0
    dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
    tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(dataFilePath)

    # TODO: add printing of loaded dataset information

    # DEBUG
    print(tmpdataKeys)

    nPatches = len(tmpdataDict['ui16coarseLimbPixels'][0])
    nSamples = nPatches * nImages

    # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
    # Initialize dataset building variables
    saveID = 0
    

    inputDataArray  = np.zeros((inputSize, nSamples), dtype=np.float32)
    labelsDataArray = np.zeros((11, nSamples), dtype=np.float32)

    for dataPair in dataFilenames:
        # Data dict for ith image
        tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(os.path.join(dataDirPath, dataPair))

        metadataDict = tmpdataDict['metadata']

        #dPosCam_TF           = np.array(metadataDict['dPosCam_TF'], dtype=np.float32)
        dAttDCM_fromTFtoCAM  = np.array(metadataDict['dAttDCM_fromTFtoCAM'], dtype=np.float32)
        dSunDir_PixCoords    = np.array(metadataDict['dSunDir_PixCoords'], dtype=np.float32)
        dLimbConicCoeffs_PixCoords = np.array(metadataDict['dLimbConic_PixCoords'], dtype=np.float32)
        #dRmoonDEM            = np.array(metadataDict['dRmoonDEM'], dtype=np.float32)

        ui16coarseLimbPixels = np.array(tmpdataDict['ui16coarseLimbPixels'], dtype=np.float32)
        ui8flattenedWindows  = np.array(tmpdataDict['ui8flattenedWindows'], dtype=np.float32)

        if id == 1:
            targetAvgRadiusInPix = np.array(metadataDict['dTargetPixAvgRadius'], dtype=np.float32)

        for sampleID in range(ui16coarseLimbPixels.shape[1]):
            # Get flattened patch
            flattenedWindow = ui8flattenedWindows[:, sampleID]
            flattenedWindSize = len(flattenedWindow)
            # Validate patch counting how many pixels are completely black or white

            pathIsValid = limbPixelExtraction_CNN_NN.IsPatchValid(flattenedWindow, lowerIntensityThr=10)

            #pathIsValid = True

            if pathIsValid:
                ptrToInput = 0
                
                # Assign flattened window to input data array
                inputDataArray[ptrToInput:ptrToInput+flattenedWindSize, saveID]  = flattenedWindow

                ptrToInput += flattenedWindSize # Update index

                #inputDataArray[49, saveID]    = dRmoonDEM

                # Assign Sun direction to input data array
                inputDataArray[ptrToInput:ptrToInput+len(dSunDir_PixCoords), saveID] = dSunDir_PixCoords

                ptrToInput += len(dSunDir_PixCoords) # Update index

                # Assign Attitude matrix as Modified Rodrigues Parameters to input data array
                tmpVal = (Rotation.from_matrix(np.array(dAttDCM_fromTFtoCAM))).as_mrp() # Convert Attitude matrix to MRP parameters
                
                inputDataArray[ptrToInput:ptrToInput+len(tmpVal), saveID] = tmpVal

                ptrToInput += len(tmpVal) # Update index

                #inputDataArray[55:58, saveID] = dPosCam_TF
                if id == 1:
                    inputDataArray[ptrToInput, saveID] = targetAvgRadiusInPix
                    ptrToInput += targetAvgRadiusInPix.size # Update index

                # Assign coarse Limb pixels to input data array
                inputDataArray[ptrToInput::, saveID] = ui16coarseLimbPixels[:, sampleID]

                # Assign labels to labels data array
                labelsDataArray[0:9, saveID] = np.ravel(dLimbConicCoeffs_PixCoords)
                labelsDataArray[9:, saveID] = ui16coarseLimbPixels[:, sampleID]

                saveID += 1


    # Shrink dataset remove entries which have not been filled due to invalid path
    print('Number of images loaded from dataset: ', nImages)
    print('Number of samples in dataset: ', nSamples)
    print('Number of removed invalid patches from validity check:', nSamples - saveID)

    inputDataArray = inputDataArray[:, 0:saveID]           
    labelsDataArray = labelsDataArray[:, 0:saveID]

    ################################################################################################

    dataDict = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
    
    # INITIALIZE DATASET OBJECT # TEMPORARY from one single dataset
    dataset = limbPixelExtraction_CNN_NN.MoonLimbPixCorrector_Dataset(dataDict)

    if exportTracedModel:
        # Save sample dataset for ONNx use
        customTorchTools.SaveTorchDataset(dataset, modelSavePath, datasetName='sampleDatasetToONNx')

    # Define the split ratio
    trainingSize = int(TRAINING_PERC * len(dataset))  
    validationSize = len(dataset) - trainingSize 

    # Split the dataset
    trainingData, validationData = torch.utils.data.random_split(dataset, [trainingSize, validationSize])

    # Define dataloaders objects
    trainingDataset   = DataLoader(trainingData, batch_size, shuffle=True)
    validationDataset = DataLoader(validationData, batch_size, shuffle=True) 

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION
    # Custom EvalLoss function: MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None)
    lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcn)

    # MODEL CLASS TYPE
    if id == 0:
        modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNN
    elif id == 1:
        modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2

    # MODEL DEFINITION
    if restartTraining:
        checkPointPath = os.path.join(modelSavePath, modelName)
        if os.path.isfile(checkPointPath):

            print('RESTART training from checkpoint: ', checkPointPath)
            modelEmpty = modelClass(outChannelsSizes, kernelSizes)
            modelCNN_NN = customTorchTools.LoadTorchModel(modelEmpty, modelName, modelSavePath)

        else:
            raise ValueError('Specified model state file not found. Check input path.')    
    else:
            modelCNN_NN = modelClass(outChannelsSizes, kernelSizes)



    # Define optimizer object specifying model instance parameters and optimizer parameters
    if optimizerID == 0:
        optimizer = torch.optim.SGD(modelCNN_NN.parameters(), lr=learnRate, momentum=momentumValue) 
    elif optimizerID == 1:
        optimizer = torch.optim.Adam(modelCNN_NN.parameters(), lr=learnRate)

    print('Using loaded dataset for training and validation: ', dataDirPath)

    # %% TRAIN and VALIDATE MODEL
    '''
    TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                              'device': GetDevice(), 
                                                                                                              'epochs': 10, 
                                                                                                              'Tensorboard':True,
                                                                                                              'saveCheckpoints':True,
                                                                                                              'checkpointsDir': './checkpoints',
                                                                                                              'modelName': 'trainedModel',
                                                                                                              'loadCheckpoint': False,
                                                                                                              'epochStart': 150}):
    '''
    (trainedModel, trainingLosses, validationLosses, inputSample) = customTorchTools.TrainAndValidateModel(dataloaderIndex, modelCNN_NN, lossFcn, optimizer, options)

    # %% Export trained model to ONNx and traced Pytorch format 
    if exportTracedModel:
        customTorchTools.ExportTorchModelToONNx(trainedModel, inputSample, onnxExportPath='./checkpoints',
                                            onnxSaveName='trainedModelONNx', modelID=options['epochStart']+options['epochs'], onnx_version=14)

        customTorchTools.SaveTorchModel(trainedModel, modelName=modelArchName+'_'+customTorchTools.AddZerosPadding(options['epochStart']+options['epochs'], 3), 
                                   saveAsTraced=True, exampleInput=inputSample)

    # Close stdout log stream
    sys.stdout.close()

# %% MAIN SCRIPT
if __name__ == '__main__':

    #for id in range(1):
    #    print('\n\n----------------------------------- RUNNING: TrainAndValidateCNN.py -----------------------------------\n')
    #    print("MAIN script operations: load dataset --> split dataset --> define dataloaders --> define model --> define loss function --> train and validate model --> export trained model\n")
    #    main(id)

    # Setup multiprocessing for training the two models in parallel

    # Use the "spawn" start method (REQUIRED by CUDA)
    multiprocessing.set_start_method('spawn')

    process1 = multiprocessing.Process(target=main, args=(0,))
    process2 = multiprocessing.Process(target=main, args=(1,))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()

    print("Training complete for both network classes. Check the logs for more information.")
