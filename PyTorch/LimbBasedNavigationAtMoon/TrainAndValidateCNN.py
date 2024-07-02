# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import sys, os, multiprocessing
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import customTorchTools # Custom torch tools
import limbPixelExtraction_CNN_NN, ModelClasses # Custom model classes
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
USE_MULTIPROCESS = False
USE_NORMALIZED_IMG = True
USE_LR_SCHEDULING = True
TRAIN_ALL = True
REGEN_DATASET = False
USE_TENSOR_LOSS_EVAL = True
MODEL_CLASS_ID = 2
MANUAL_RUN = True # Uses MODEL_CLASS_ID to run a specific model

USE_BATCH_NORM = True

def main(idSession:int):

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [256, 128, 256, 64] 

    if MODEL_CLASS_ID == 2 or idSession == 2:
        outChannelsSizes = [128, 64, 64+8, 32, 16] 
    elif MODEL_CLASS_ID == 3 or idSession == 3:
        outChannelsSizes = [256, 128, 64, 32]

    kernelSizes = [3, 3]
    initialLearnRate = 1E-3
    momentumValue = 0.9

    #TODO: add log and printing of settings of optimizer for each epoch. Reduce the training loss value printings

    LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch, 
                  # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
    # Loss function parameters
    params = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 0}

    optimizerID = 0 # 0
    UseMaxPooling = True

    device = customTorchTools.GetDevice()

    exportTracedModel = True    
    tracedModelSavePath = 'tracedModelsArchive' 

    # Options to restart training from checkpoint
    if idSession == 0:
        ID = 0
        runID = "{num:04d}".format(num=ID)
        #modelSavePath = './checkpoints/HorizonPixCorrector_CNNv1_run3'
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv1max_largerCNN_run' + runID
        datasetSavePath = './datasets/HorizonPixCorrectorV1'
        tensorboardLogDir = './tensorboardLogs/tensorboardLog_v1max_largerCNN_run' + runID
        tensorBoardPortNum = 6008
        
        modelArchName = 'HorizonPixCorrector_CNNv1max_largerCNN_run' + runID
        inputSize = 56 # TODO: update this according to new model
        numOfEpochs = 15


    elif idSession == 1:
        ID = 0
        runID = "{num:04d}".format(num=ID)
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2max_largerCNN_run' + runID
        datasetSavePath = './datasets/HorizonPixCorrectorV2'
        tensorboardLogDir = './tensorboardLogs/tensorboardLog_v2max_largerCNN_run'   + runID
        tensorBoardPortNum = 6009

        modelArchName = 'HorizonPixCorrector_CNNv2max_largerCNN_run' + runID
        inputSize = 57 # TODO: update this according to new model
        numOfEpochs = 15

    elif idSession == 2:
        ID = 1
        runID = "{num:04d}".format(num=ID) 
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run' + runID
        datasetSavePath = './datasets/HorizonPixCorrectorV3'
        tensorboardLogDir = './tensorboardLogs/tensorboardLog_v3max_largerCNNdeeperNN_run'   + runID
        tensorBoardPortNum = 6010

        modelArchName = 'HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run' + runID
        inputSize = 57 # TODO: update this according to new model
        numOfEpochs = 20


    elif idSession == 3:
        ID = 0
        runID = "{num:04d}".format(num=ID)
        modelSavePath = './checkpoints/HorizonPixCorrector_deepNNv4_run' + runID
        datasetSavePath = './datasets/HorizonPixCorrectorV4'
        tensorboardLogDir = './tensorboardLogs/tensorboardLog_deepNNv4_run'   + runID
        tensorBoardPortNum = 6011

        modelArchName = 'HorizonPixCorrector_deepNNv4_run' + runID
        inputSize = 57 # TODO: update this according to new model
        numOfEpochs = 15


    if USE_MULTIPROCESS == True:
        # Start tensorboard session

        sys.stdout = open("./multiprocessShellLog/stdout_log_" + modelArchName + '.txt', 'w') # Redirect print outputs


    options = {'taskType': 'regression', 
               'device': device, 
               'epochs': numOfEpochs, 
               'Tensorboard':True,
               'saveCheckpoints':True,
               'checkpointsOutDir': modelSavePath,
               'modelName': modelArchName,
               'loadCheckpoint': False,
               'checkpointsInDir': modelSavePath,
               'lossLogName': 'LossOutOfPatch_MoonHorizonExtraction',
               'logDirectory': tensorboardLogDir,
               'epochStart': 0,
               'tensorBoardPortNum': tensorBoardPortNum}



    if options['epochStart'] == 0:
        restartTraining = False
    else:
        restartTraining = True
        # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
        with os.scandir(modelSavePath) as it:
            modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]


    # %% GENERATE OR LOAD TORCH DATASET

    dataPath = os.path.join('/home/peterc/devDir/MATLABcodes/syntheticRenderings/Datapairs')
    dataPath = os.path.join('/home/peterc/devDir/lumio/lumio-prototype/src/dataset_gen/Datapairs')
    dirNamesRoot = os.listdir(dataPath)

    #os.scandir(dataDirPath)
    #with os.scandir(modelSavePath) as it:
    #    modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
    #    modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]

    assert(len(dirNamesRoot) >= 2)

    # Select one of the available datapairs folders (each corresponding to a labels generation pipeline output)
    datasetID = [5, 7] # TRAINING and VALIDATION datasets ID in folder (in this order)

    assert(len(datasetID) == 2)

    TrainingDatasetTag = dirNamesRoot[datasetID[0]]
    ValidationDatasetTag = dirNamesRoot[datasetID[1]]

    datasetNotFound = not(os.path.isfile(os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt'))) or not((os.path.isfile(os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt'))))

    if REGEN_DATASET == True or datasetNotFound:
        # REGENERATE the dataset
        # TODO: add printing of loaded dataset information

        for idDataset in range(len(datasetID)):

            if idDataset == 0:
                print('Generating training dataset from: ', dirNamesRoot[datasetID[idDataset]])
            elif idDataset == 1:
                print('Generating validation dataset from: ', dirNamesRoot[datasetID[idDataset]])

            # Get images and labels from IDth dataset
            dataDirPath = os.path.join(dataPath, dirNamesRoot[datasetID[idDataset]])
            dataFilenames = os.listdir(dataDirPath)
            nImages = len(dataFilenames)

            print('Found number of images:', nImages)

            # DEBUG
            #print(dataFilenames)

            # Get nPatches from the first datapairs files
            dataFileID = 0
            dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
            #print('Loading data from:', dataFilePath)
            tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(dataFilePath)
            print('All data loaded correctly --> DATASET PREPARATION BEGIN')

            # DEBUG
            #print(tmpdataKeys)

            nPatches = len(tmpdataDict['ui16coarseLimbPixels'][0])
            nSamples = nPatches * nImages

            # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
            # Initialize dataset building variables
            saveID = 0

            inputDataArray  = np.zeros((inputSize, nSamples), dtype=np.float32)
            labelsDataArray = np.zeros((14, nSamples), dtype=np.float32)

            imgID = 0
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

                dTruePixOnConic = np.array(tmpdataDict['dTruePixOnConic'], dtype=np.float32)

                invalidPatchesToCheck = {'ID': [], 'flattenedPatch': []}

                if idSession == 1 or idSession == 2 or idSession == 3:
                    targetAvgRadiusInPix = np.array(metadataDict['dTargetPixAvgRadius'], dtype=np.float32)

                if USE_NORMALIZED_IMG:
                    normalizationCoeff = 255.0
                else:
                    normalizationCoeff = 1.0

                for sampleID in range(ui16coarseLimbPixels.shape[1]):

                    print("\tProcessing samples: {current:8.0f}/{total:8.0f} of image {currentImgID:5.0f}/{totalImgs:5.0f}".format(current=sampleID+1, 
                    total=ui16coarseLimbPixels.shape[1], currentImgID=imgID+1, totalImgs=nImages), end="\r")

                    # Get flattened patch
                    flattenedWindow = ui8flattenedWindows[:, sampleID]
                    flattenedWindSize = len(flattenedWindow)
                    # Validate patch counting how many pixels are completely black or white

                    #pathIsValid = limbPixelExtraction_CNN_NN.IsPatchValid(flattenedWindow, lowerIntensityThr=10)

                    pathIsValid = True

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
                        if idSession == 1 or idSession == 2 or idSession == 3:
                            inputDataArray[ptrToInput, saveID] = targetAvgRadiusInPix
                            ptrToInput += targetAvgRadiusInPix.size # Update index

                        # Assign coarse Limb pixels to input data array
                        inputDataArray[ptrToInput::, saveID] = ui16coarseLimbPixels[:, sampleID]

                        # Assign labels to labels data array (DO NOT CHANGE ORDER OF VALUES)
                        labelsDataArray[0:9, saveID] = np.ravel(dLimbConicCoeffs_PixCoords)
                        labelsDataArray[9:11, saveID] = ui16coarseLimbPixels[:, sampleID]
                        labelsDataArray[11, saveID] = centreBaseCost2[sampleID]
                        labelsDataArray[12:14, saveID] = dTruePixOnConic[:, sampleID]

                        saveID += 1
                    else:
                        # Save invalid patches to check
                        invalidPatchesToCheck['ID'].append(sampleID)
                        invalidPatchesToCheck['flattenedPatch'].append(flattenedWindow.tolist())
                imgID += 1

            # Save json with invalid patches to check
            #if idSession == 0:
            #    fileName = './invalidPatchesToCheck_Training_.json'
            #elif idSession == 1:
            #    fileName = './invalidPatchesToCheck_Validation.json'
            #with open(os.path.join(fileName), 'w') as fileCheck:
            #    invalidPatchesToCheck_string = json.dumps(invalidPatchesToCheck)
            #    json.dump(invalidPatchesToCheck_string, fileCheck)
            #    fileCheck.close()

            # Shrink dataset remove entries which have not been filled due to invalid path
            print('\n')
            print('\tNumber of images loaded from dataset: ', nImages)
            print('\tNumber of samples in dataset: ', nSamples)
            print('\tNumber of removed invalid patches from validity check:', nSamples - saveID)

            inputDataArray = inputDataArray[:, 0:saveID]           
            labelsDataArray = labelsDataArray[:, 0:saveID]

            if idDataset == 0:
                dataDictTraining = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
            elif idDataset == 1:   
                dataDictValidation = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}

            print('DATASET PREPARATION COMPLETED.')

        # INITIALIZE DATASET OBJECT # TEMPORARY from one single dataset
        datasetTraining = limbPixelExtraction_CNN_NN.MoonLimbPixCorrector_Dataset(dataDictTraining)
        datasetValidation = limbPixelExtraction_CNN_NN.MoonLimbPixCorrector_Dataset(dataDictValidation)

        # Save dataset as torch dataset object for future use
        if not os.path.exists(datasetSavePath):
            os.makedirs(datasetSavePath)
        
        customTorchTools.SaveTorchDataset(datasetTraining, datasetSavePath, datasetName='TrainingDataset_'+TrainingDatasetTag)
        customTorchTools.SaveTorchDataset(datasetValidation, datasetSavePath, datasetName='ValidationDataset_'+ValidationDatasetTag)
    else:

        if not(os.path.isfile(os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt'))) or not((os.path.isfile(os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt')))):
            raise ImportError('Dataset files not found. Set REGEN_DATASET to True to regenerate the dataset torch saves.')
        
        # LOAD PRE-GENERATED DATASETS
        trainingDatasetPath = os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt')
        validationDatasetPath = os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt')
        
        print('Loading training and validation datasets from: \n\t', trainingDatasetPath, '\n\t', validationDatasetPath)

        datasetTraining = customTorchTools.LoadTorchDataset(datasetSavePath, datasetName='TrainingDataset_'+TrainingDatasetTag)
        datasetValidation = customTorchTools.LoadTorchDataset(datasetSavePath, datasetName='ValidationDataset_'+ValidationDatasetTag)
    ################################################################################################

    # SUPERSEDED CODE --> move this to function for dataset splitting (add rng seed for reproducibility)
    # Define the split ratio
    #trainingSize = int(TRAINING_PERC * len(dataset))  
    #validationSize = len(dataset) - trainingSize 

    # Split the dataset
    #trainingData, validationData = torch.utils.data.random_split(dataset, [trainingSize, validationSize])

    # Define dataloaders objects
    trainingDataset   = DataLoader(datasetTraining, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    validationDataset = DataLoader(datasetValidation, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}

    # LOSS FUNCTION DEFINITION

    if LOSS_TYPE == 0:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcn, params)
    elif LOSS_TYPE == 1:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm, params)
    elif LOSS_TYPE == 2:
        if USE_TENSOR_LOSS_EVAL:
            lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm_asTensor, params)
        else:
            lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm, params)
    elif LOSS_TYPE == 3:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_PolarNdirectionDistanceWithOutOfPatch_asTensor, params)

    elif LOSS_TYPE == 4:
        lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor, params)

    # MODEL CLASS TYPE
    if UseMaxPooling == False:
        if idSession == 0:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1avg
        elif idSession == 1:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2avg
    else:
        if idSession == 0:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1max
        elif idSession == 1:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2max
        elif idSession == 2:
            modelClass = ModelClasses.HorizonExtractionEnhancerCNNv3maxDeeper
        elif idSession == 3:
            modelClass = ModelClasses.HorizonExtractionEnhancer_FullyConnNNv4        
    

    # MODEL DEFINITION
    if restartTraining:
        checkPointPath = os.path.join(modelSavePath, modelName)
        if os.path.isfile(checkPointPath):

            print('RESTART training from checkpoint: ', checkPointPath)
            #modelEmpty = modelClass(outChannelsSizes, kernelSizes)
            modelCNN_NN = customTorchTools.LoadTorchModel(None, modelName, modelSavePath, loadAsTraced=True).to(device=device)
            #modelCNN_NN = customTorchTools.LoadTorchModel(modelCNN_NN, modelName)

        else:
            raise ValueError('Specified model state file not found. Check input path.')    
    else:
        if idSession == 0 or idSession == 1:
            modelCNN_NN = modelClass(outChannelsSizes, kernelSizes).to(device=device)
        elif idSession == 2:
            modelCNN_NN = modelClass(outChannelsSizes, kernelSizes, USE_BATCH_NORM).to(device=device)
        elif idSession == 3:
            modelCNN_NN = modelClass(outChannelsSizes).to(device=device)


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
        optimizer = torch.optim.SGD(modelCNN_NN.parameters(), lr=initialLearnRate, momentum=momentumValue) 
    
    elif optimizerID == 1:
        optimizer = torch.optim.Adam(modelCNN_NN.parameters(), lr=initialLearnRate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=1E-6, amsgrad=False, foreach=None, fused=True)

    if USE_LR_SCHEDULING:
        #optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, threshold_mode='rel', cooldown=1, min_lr=1E-12, eps=1e-08)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=options['epochStart']-1)
        options['lr_scheduler'] = lr_scheduler


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

        customTorchTools.SaveTorchModel(trainedModel, modelName=os.path.join(tracedModelSavePath, modelArchName+'_'+str(customTorchTools.AddZerosPadding(options['epochStart']+options['epochs'], 3) )), 
                                   saveAsTraced=True, exampleInput=inputSample)

    # Close stdout log stream
    if USE_MULTIPROCESS == True:
        sys.stdout.close()

# %% MAIN SCRIPT
if __name__ == '__main__':

    # Stop any running tensorboard session before starting a new one
    customTorchTools.KillTensorboard()

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
    elif USE_MULTIPROCESS == False and MANUAL_RUN == False:

        for idSession in range(4 if TRAIN_ALL else 1):
            print('\n\n----------------------------------- RUNNING: TrainAndValidateCNN.py -----------------------------------\n')
            print("MAIN script operations: load dataset --> split dataset --> define dataloaders --> define model --> define loss function --> train and validate model --> export trained model\n")
            main(idSession)
    else:
        main(MODEL_CLASS_ID)
