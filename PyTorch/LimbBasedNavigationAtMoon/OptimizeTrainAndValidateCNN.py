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
from sklearn import preprocessing # Import scikit-learn for dataset preparation


import torch, mlflow, optuna
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
MODEL_CLASS_ID = 4
MANUAL_RUN = True # Uses MODEL_CLASS_ID to run a specific model
LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch,            
            # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
RUN_ID = 1
USE_BATCH_NORM = True
FULLY_PARAMETRIC = True

# SETTINGS and PARAMETERS 
batch_size = 16*4 # Defines batch size in dataset
#outChannelsSizes = [16, 32, 75, 15] 
    
#outChannelsSizes = [2056, 1024, 512, 512, 128, 64]

kernelSizes = [3, 3]
#initialLearnRate = 1E-1
momentumValue = 0.6

#TODO: add log and printing of settings of optimizer for each epoch. Reduce the training loss value printings

# Loss function parameters
params = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 1}

optimizerID = 1 # 0: SGD, 1: Adam
UseMaxPooling = True

device = customTorchTools.GetDevice()

exportTracedModel = True    
tracedModelSavePath = 'tracedModelsArchive' 

modelSavePath = './checkpoints/HorizonExtractionEnhancer_deepNNv8_fullyParam' 
datasetSavePath = './datasets/HorizonExtractionEnhancer_deepNNv8_fullyParam'
#tensorboardLogDir = './tensorboardLogs/tensorboardLog_ShortCNNv6maxDeeper_run'   + runID
#tensorBoardPortNum = 6012

modelArchName = 'HorizonExtractionEnhancer_deepNNv8_fullyParam' 
inputSize = 58 # TODO: update this according to new model
numOfEpochs = 4

EPOCH_START = 0

options = {'taskType': 'regression', 
           'device': device, 
           'epochs': numOfEpochs, 
           'saveCheckpoints':True,
           'checkpointsOutDir': modelSavePath,
           'modelName': modelArchName,
           'loadCheckpoint': False,
           'checkpointsInDir': modelSavePath,
           'lossLogName': 'MSE_OutOfPatch',
           #'logDirectory': tensorboardLogDir,
           'epochStart': EPOCH_START,
           #'tensorBoardPortNum': tensorBoardPortNum
           }


# %% GENERATE OR LOAD TORCH DATASET

dataPath = os.path.join('/home/peterc/devDir/MATLABcodes/syntheticRenderings/Datapairs')
dataPath = os.path.join('/home/peterc/devDir/lumio/lumio-prototype/src/dataset_gen/Datapairs')
#dirNamesRoot = os.listdir(dataPath)

with os.scandir(dataPath) as it:
    modelNamesWithID = [(entry.name, entry.name[3:6]) for entry in it if entry.is_dir()]
    dirNamesRootTuples = sorted(modelNamesWithID, key=lambda x: int(x[1]))

dirNamesRoot = [stringVal for stringVal, _ in dirNamesRootTuples]

assert(len(dirNamesRoot) >= 2)

# Select one of the available datapairs folders (each corresponding to a labels generation pipeline output)
datasetID = [6, 7] # TRAINING and VALIDATION datasets ID in folder (in this order)


assert(len(datasetID) == 2)

TrainingDatasetTag = 'ID_006_Datapairs_TrainingSet_20240705'
ValidationDatasetTag = 'ID_007_Datapairs_ValidationSet_20240705'

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
            dAttDCM_fromTFtoCAM   = np.array(metadataDict['dAttDCM_fromTFtoCAM'], dtype=np.float32)
            #dSunDir_PixCoords    = np.array(metadataDict['dSunDir_PixCoords'], dtype=np.float32)
            dSunDirAngle          = np.array(metadataDict['dSunAngle'], dtype=np.float32)
            dLimbConicCoeffs_PixCoords = np.array(metadataDict['dLimbConic_PixCoords'], dtype=np.float32)
            #dRmoonDEM            = np.array(metadataDict['dRmoonDEM'], dtype=np.float32)
            dConicPixCente        = np.array(metadataDict['dConicPixCentre'], dtype=np.float32)
            ui16coarseLimbPixels = np.array(tmpdataDict['ui16coarseLimbPixels'], dtype=np.float32)
            ui8flattenedWindows  = np.array(tmpdataDict['ui8flattenedWindows'], dtype=np.float32)
            centreBaseCost2 = np.array(tmpdataDict['dPatchesCentreBaseCost2'], dtype=np.float32)
            try:
                dTruePixOnConic = np.array(tmpdataDict['dTruePixOnConic'], dtype=np.float32)
            except:
                dTruePixOnConic = np.array(tmpdataDict['truePixOnConic'], dtype=np.float32)
            #invalidPatchesToCheck = {'ID': [], 'flattenedPatch': []}
            dConicAvgRadiusInPix = np.array(metadataDict['dTargetPixAvgRadius'], dtype=np.float32)
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
                    inputDataArray[ptrToInput, saveID] = dSunDirAngle
                    ptrToInput += 1 # Update index

                    # Assign Attitude matrix as Modified Rodrigues Parameters to input data array
                    tmpVal = (Rotation.from_matrix(np.array(dAttDCM_fromTFtoCAM))).as_mrp() # Convert Attitude matrix to MRP parameters
                    inputDataArray[ptrToInput:ptrToInput+len(tmpVal), saveID] = tmpVal
                    ptrToInput += len(tmpVal) # Update index

                    #inputDataArray[55:58, saveID] = dPosCam_TF
                    inputDataArray[ptrToInput, saveID] = dConicAvgRadiusInPix
                    ptrToInput += dConicAvgRadiusInPix.size # Update index
                    inputDataArray[ptrToInput:ptrToInput+2, saveID] = dConicPixCente
                    ptrToInput += dConicPixCente.size # Update index

                    # Assign coarse Limb pixels to input data array
                    inputDataArray[ptrToInput:ptrToInput+len(ui16coarseLimbPixels[:, sampleID]), saveID] = ui16coarseLimbPixels[:, sampleID]
                    # Assign labels to labels data array (DO NOT CHANGE ORDER OF VALUES)
                    labelsDataArray[0:9, saveID] = np.ravel(dLimbConicCoeffs_PixCoords)
                    labelsDataArray[9:11, saveID] = ui16coarseLimbPixels[:, sampleID]
                    labelsDataArray[11, saveID] = centreBaseCost2[sampleID]
                    labelsDataArray[12:14, saveID] = dTruePixOnConic[:, sampleID]
                    
                    saveID += 1
                #else:
                    # Save invalid patches to check
                    #invalidPatchesToCheck['ID'].append(sampleID)
                    #invalidPatchesToCheck['flattenedPatch'].append(flattenedWindow.tolist())
            imgID += 1

        # Save json with invalid patches to check
        #if idModelClass == 0:
        #    fileName = './invalidPatchesToCheck_Training_.json'
        #elif idModelClass == 1:
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
        # Apply standardization to input data # TODO: check if T is necessary
        #ACHTUNG: image may not be standardized? --> TBC
        inputDataArray[flattenedWindSize:, :] = preprocessing.StandardScaler().fit_transform(inputDataArray[flattenedWindSize:, :].T).T
        if idDataset == 0:
            dataDictTraining = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
        elif idDataset == 1:   
            dataDictValidation = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
        print('DATASET PREPARATION COMPLETED.')

    # INITIALIZE DATASET OBJECT # TEMPORARY from one single dataset
    datasetTraining = datasetPreparation.MoonLimbPixCorrector_Dataset(dataDictTraining)
    datasetValidation = datasetPreparation.MoonLimbPixCorrector_Dataset(dataDictValidation)
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
else:
    raise ValueError('Loss function ID not found.')

def objective(trial):

    device = customTorchTools.GetDevice()

    # START MLFLOW RUN LOGGING
    modelSavePath = './checkpoints/HorizonExtractionEnhancer_deepNNv8' 
    modelArchName = 'HorizonExtractionEnhancer_deepNNv8' 

    with mlflow.start_run() as mlflow_run:
        run_id = mlflow_run.info.run_id
        run_name = mlflow_run.info.run_name
        # Add run_id to save names

        modelSavePath += ('_' + run_name)
        modelArchName += ('_' + run_name)

        options['modelName'] = modelArchName
        options['checkpointsOutDir'] = modelSavePath

        # MODEL DEFINITION

        # Model layers width
        outChannelsSizes = []

        if FULLY_PARAMETRIC:
            num_layers = trial.suggest_int('num_layers', 2, 25)
            maxNodes = 4096
            modelClass = ModelClasses.HorizonExtractionEnhancer_deepNNv8_fullyParametric
        else:
            num_layers = 6
            maxNodes = 8192
            modelClass = ModelClasses.HorizonExtractionEnhancer_deepNNv8


        for i in range(num_layers):
            outChannelsSizes.append(trial.suggest_int(f'DenseL{i}', 32, maxNodes))
            mlflow.log_param(f'DenseL{i}', outChannelsSizes[-1])

        parametersConfig = {'useBatchNorm': True, 'alphaDropCoeff': 0.1, 'LinearInputSize': 58, 
                    'outChannelsSizes': outChannelsSizes}
        

        model = modelClass(parametersConfig).to(device=device)
    
        mlflow.log_params(parametersConfig)
        mlflow.log_param('NumOfHiddenLayers', num_layers-1)
        mlflow.log_params(options)

        # Define optimizer object specifying model instance parameters and optimizer parameters
        initialLearnRate = trial.suggest_float('lr', 1e-8, 1e-3, log=True) # NOTE: What are the inputs to suggest_float?

        optimizer = torch.optim.Adam(model.parameters(), lr=initialLearnRate, betas=(0.9, 0.999), 
                                    eps=1e-08, weight_decay=1E-6, amsgrad=False, foreach=None, fused=True)
    
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
    
        exponentialDecayGamma = 0.9
    
        for name, param in model.named_parameters():
            mlflow.log_param(name, list(param.size()))
        
        if USE_LR_SCHEDULING:
            #optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, threshold_mode='rel', cooldown=1, min_lr=1E-12, eps=1e-08)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponentialDecayGamma, last_epoch=options['epochStart']-1)
            options['lr_scheduler'] = lr_scheduler
    
        # Log model parameters
        mlflow.log_param('optimizer ID', 1)
        mlflow.log_param('learning_rate', initialLearnRate)
        mlflow.log_param('ExponentialDecayGamma', exponentialDecayGamma)
    
        # %% TRAIN and VALIDATE MODEL
        (trainedModel, trainingLosses, validationLosses, inputSample) = customTorchTools.TrainAndValidateModel(dataloaderIndex, model, lossFcn, optimizer, options)

        return validationLosses[-1] # Return last validation loss value


# %% MAIN SCRIPT
if __name__ == '__main__':
    
    # Set mlflow experiment
    #mlflow.set_experiment("HorizonEnhancerCNN_OptimizationRuns")
    #mlflow.set_experiment("HorizonEnhancerCNN_OptimizationRuns_TrainValidOnRandomCloud")
    mlflow.set_experiment("HorizonEnhancerCNN_OptunaRuns_deepNNv8_fullyParametric")
    # Stop any running tensorboard session before starting a new one
    customTorchTools.KillTensorboard()

    # Setup multiprocessing for training the two models in parallel
    if USE_MULTIPROCESS == True:

        # Use the "spawn" start method (REQUIRED by CUDA)
        multiprocessing.set_start_method('spawn')
        #process1 = multiprocessing.Process(target=main, args=(10,1, 4))
        #process2 = multiprocessing.Process(target=main, args=(10,1, 4))

        # Start the processes
        #process1.start()
        #process2.start()

        # Wait for both processes to finish
        #process1.join()
        #process2.join()

        print("Training complete for both network classes. Check the logs for more information.")
    elif USE_MULTIPROCESS == False and MANUAL_RUN == False:
        
        modelClassesToTrain = [0, 1, 2, 3]
        idLossType = 4
        idRun = 0

        for idModelClass in modelClassesToTrain:
            print('\n\n----------------------------------- RUNNING: TrainAndValidateCNN.py -----------------------------------\n')
            print("MAIN script operations: load dataset --> split dataset --> define dataloaders --> define model --> define loss function --> train and validate model --> export trained model\n")
            #main(idRun, idModelClass, idLossType)
    
    else:
        idLossType = LOSS_TYPE
        RUN_OPTUNA_STUDY = True

        if RUN_OPTUNA_STUDY:
            # %% Optuna study configuration
            optunaStudyObj = optuna.create_study(study_name='HorizonEnhancerCNN_HyperOptimization_deepNNv8_fullyParametric', direction='minimize',
                                                 pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, 
                                                                                               min_early_stopping_rate=0))

            # %% Optuna optimization
            optunaStudyObj.optimize(objective, n_trials=500, timeout=7*3600)

            # Print the best trial
            print('Number of finished trials:', len(optunaStudyObj.trials)) # Get number of finished trials
            print('Best trial:')

            trial = optunaStudyObj.best_trial # Get the best trial from study object
            print('  Value: {:.4f}'.format(trial.value)) # Loss function value for the best trial

            # Print parameters of the best trial
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))
                
        else:
            idRun = RUN_ID
            #main(idRun, MODEL_CLASS_ID, idLossType)
