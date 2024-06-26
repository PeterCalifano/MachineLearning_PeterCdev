# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import sys, os, multiprocessing
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import customTorchTools # Custom torch tools
import limbPixelExtraction_CNN_NN
import datasetPreparation

import torch
import datetime, json
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim

# EXECUTION MODE
USE_MULTIPROCESS = True
USE_NORMALIZED_IMG = True

def main(id):

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    TRAINING_PERC = 0.85
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [256, 128, 75, 50] 
    kernelSizes = [3, 3]
    learnRate = 1E-9
    momentumValue = 0.001

    LOSS_TYPE = 2 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch
    # Loss function parameters
    params = {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 1}

    optimizerID = 1 # 0
    UseMaxPooling = True

    device = customTorchTools.GetDevice()

    exportTracedModel = True    
    tracedModelSavePath = 'tracedModelsArchive' 

    # Options to restart training from checkpoint
    if id == 0:
        runID = str(3)
        #modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2_run3'
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv1max_largerCNN_run' + runID

        tensorboardLogDir = './tensorboardLogs/tensorboardLog_v1max_largerCNN_run' + runID
        tensorBoardPortNum = 6006
        
        modelArchName = 'HorizonPixCorrector_CNNv1max_largerCNN_run' + runID
        inputSize = 56 # TODO: update this according to new model


    elif id == 1:
        runID = str(3)
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2max_largerCNN_run' + runID

        tensorboardLogDir = './tensorboardLogs/tensorboardLog_v2max_largerCNN_run'   + runID
        tensorBoardPortNum = 6007

        modelArchName = 'HorizonPixCorrector_CNNv2max_largerCNN_run' + runID
        inputSize = 57 # TODO: update this according to new model



    if USE_MULTIPROCESS == True:
        # Start tensorboard session

        sys.stdout = open("./multiprocessShellLog/stdout_log_" + modelArchName + '.txt', 'w') # Redirect print outputs


    options = {'taskType': 'regression', 
               'device': device, 
               'epochs': 5, 
               'Tensorboard':True,
               'saveCheckpoints':True,
               'checkpointsOutDir': modelSavePath,
               'modelName': modelArchName,
               'loadCheckpoint': False,
               'checkpointsInDir': modelSavePath,
               'lossLogName': 'LossOutOfPatch_MoonHorizonExtraction',
               'logDirectory': tensorboardLogDir,
               'epochStart': 5,
               'tensorBoardPortNum': tensorBoardPortNum}



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
    datasetID = 0 # ACHTUNG! paths from listdir are randomly ordered! --> TODO: modify
    dataDirPath = os.path.join(dataPath, dirNamesRoot[datasetID])

    dataFilenames = os.listdir(dataDirPath)

    nImages = len(dataFilenames)

    # DEBUG
    print('Found images:', nImages)
    print(dataFilenames)

    # Get nPatches from the first datapairs files
    dataFileID = 0
    dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
    print('Loading data from:', dataFilePath)
    tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(dataFilePath)
    print('All data loaded correctly.')

    # TODO: add printing of loaded dataset information

    # DEBUG
    print(tmpdataKeys)

    nPatches = len(tmpdataDict['ui16coarseLimbPixels'][0])
    nSamples = nPatches * nImages

    # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
    # Initialize dataset building variables
    saveID = 0
    

    inputDataArray  = np.zeros((inputSize, nSamples), dtype=np.float32)
    labelsDataArray = np.zeros((12, nSamples), dtype=np.float32)

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

        centreBaseCost2 = np.array(tmpdataDict['dPatchesCentreBaseCost2'], dtype=np.float32)

        invalidPatchesToCheck = {'ID': [], 'flattenedPatch': []}


        if id == 1:
            targetAvgRadiusInPix = np.array(metadataDict['dTargetPixAvgRadius'], dtype=np.float32)

        if USE_NORMALIZED_IMG:
            normalizationCoeff = 255.0
        else:
            normalizationCoeff = 1.0
        
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
                inputDataArray[ptrToInput:ptrToInput+flattenedWindSize, saveID]  = flattenedWindow/normalizationCoeff

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
                labelsDataArray[9:11, saveID] = ui16coarseLimbPixels[:, sampleID]
                labelsDataArray[11, saveID] = centreBaseCost2[sampleID]

                saveID += 1
            else:
                # Save invalid patches to check
                invalidPatchesToCheck['ID'].append(sampleID)
                invalidPatchesToCheck['flattenedPatch'].append(flattenedWindow.tolist())
                
                

    # Save json with invalid patches to check
    with open(os.path.join('./invalidPatchesToCheck.json'), 'w') as fileCheck:
        invalidPatchesToCheck_string = json.dumps(invalidPatchesToCheck)
        json.dump(invalidPatchesToCheck_string, fileCheck)
        fileCheck.close()

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
    trainingDataset   = DataLoader(trainingData, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    validationDataset = DataLoader(validationData, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION

    if LOSS_TYPE == 0:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcn, params)
    elif LOSS_TYPE == 1:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm, params)
    elif LOSS_TYPE == 2:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm, params)
        

    # MODEL CLASS TYPE
    if UseMaxPooling == False:
        if id == 0:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1avg
        elif id == 1:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2avg
    else:
        if id == 0:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1max
        elif id == 1:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2max

    # MODEL DEFINITION
    if restartTraining:
        checkPointPath = os.path.join(modelSavePath, modelName)
        if os.path.isfile(checkPointPath):

            print('RESTART training from checkpoint: ', checkPointPath)
            #modelEmpty = modelClass(outChannelsSizes, kernelSizes)
            modelCNN_NN = customTorchTools.LoadTorchModel(None, modelName, modelSavePath, loadAsTraced=True)
            #modelCNN_NN = customTorchTools.LoadTorchModel(modelCNN_NN, modelName)

        else:
            raise ValueError('Specified model state file not found. Check input path.')    
    else:
            modelCNN_NN = modelClass(outChannelsSizes, kernelSizes)


    # ######### TEST DEBUG ############
    #testPrediction = modelCNN_NN(torch.tensor(inputDataArray[0:, 0]))

    ############    END TEST DEBUG    ############
    #exit()

    #try:
    #    modelCNN_NN = torch.compile(modelCNN_NN) # ACHTUNG: compile is not compatible with jit.trace required to save traced model.
    #    print('Model compiled successfully.')
    #except:
    #    print('Model compilation failed. Using eager mode.')

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
       #customTorchTools.ExportTorchModelToONNx(trainedModel, inputSample, onnxExportPath='./checkpoints',
       #                                    onnxSaveName='trainedModelONNx', modelID=options['epochStart']+options['epochs'], onnx_version=14)

        customTorchTools.SaveTorchModel(trainedModel, modelName=os.path.join(tracedModelSavePath, modelArchName+'_'+customTorchTools.AddZerosPadding(options['epochStart']+options['epochs']), 3), 
                                   saveAsTraced=True, exampleInput=inputSample)

    # Close stdout log stream
    if USE_MULTIPROCESS == True:
        sys.stdout.close()

# %% MAIN SCRIPT
if __name__ == '__main__':

    # Setup multiprocessing for training the two models in parallel
    if USE_MULTIPROCESS == True:
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
    else:
        for id in range(2):
            print('\n\n----------------------------------- RUNNING: TrainAndValidateCNN.py -----------------------------------\n')
            print("MAIN script operations: load dataset --> split dataset --> define dataloaders --> define model --> define loss function --> train and validate model --> export trained model\n")
            main(id)