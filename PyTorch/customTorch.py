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
import sys, os, signal
import subprocess
import psutil
import inspect
import onnx

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
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
    #print(f"Using {device} device")
    return device

# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

def TrainModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, optimizer, device=GetDevice(), taskType:str='classification'):

    size=len(dataloader.dataset) # Get size of dataloader dataset object
    model.train() # Set model instance in training mode ("informing" backend that the training is going to start)

    for batchCounter, (X, Y) in enumerate(dataloader): # Recall that enumerate gives directly both ID and value in iterable object

        # Get input and labels and move to target device memory
        X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device

        # Perform FORWARD PASS to get predictions
        predVal = model(X) # Evaluate model at input
        trainLoss = lossFcn(predVal, Y) # Evaluate loss function to get loss value (this returns loss function instance, not a value)

        # Perform BACKWARD PASS to update parameters
        trainLoss.backward()  # Compute gradients
        optimizer.step()      # Apply gradients from the loss
        optimizer.zero_grad() # Reset gradients for next iteration

        if batchCounter % 100 == 0: # Print loss value every 100 steps
            trainLoss, currentStep = trainLoss.item(), (batchCounter + 1) * len(X)
            print(f"Training loss value: {trainLoss:>7f}  [{currentStep:>5d}/{size:>5d}]")

    return trainLoss
    
# %% Function to validate model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

def ValidateModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, device=GetDevice(), taskType:str='classification'):
    # Auxiliary variables
    size = len(dataloader.dataset) 
    numberOfBatches = len(dataloader)

    model.eval() # Set the model in evaluation mode
    validationLoss = 0 # Accumulation variables

    # Initialize variables based on task type
    if taskType.lower() == 'classification': 
        correctOuputs = 0

    elif taskType.lower() == 'regression':
        avgRelAccuracy = 0.0
        avgAbsAccuracy = 0.0

    elif taskType.lower() == 'custom':
        print('TODO')

    with torch.no_grad(): # Tell torch that gradients are not required
        for X,Y in dataloader:
            # Get input and labels and move to target device memory
            X, Y = X.to(device), Y.to(device)  

            # Perform FORWARD PASS
            predVal = model(X) # Evaluate model at input
            validationLoss += lossFcn(predVal, Y).item() # Evaluate loss function and accumulate

            if taskType.lower() == 'classification': 
                # Determine if prediction is correct and accumulate
                # Explanation: get largest output logit (the predicted class) and compare to Y. 
                # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
                correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item() 

            #elif taskType.lower() == 'regression':
            #    #print('TODO')
#
            #elif taskType.lower() == 'custom':
            #    print('TODO')


    if taskType.lower() == 'classification': 
        validationLoss /= numberOfBatches # Compute batch size normalized loss value
        correctOuputs /= size # Compute percentage of correct classifications over batch size
        print(f"\n VALIDATION ((Classification) Accuracy: {(100*correctOuputs):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'regression':
        #print('TODO')
        validationLoss /= numberOfBatches
        print(f"\n VALIDATION (Regression) Avg loss: {validationLoss:>0.1f}\n")
        #print(f"Validation (Regression): \n Avg absolute accuracy: {avgAbsAccuracy:>0.1f}, Avg relative accuracy: {(100*avgRelAccuracy):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'custom':
        print('TODO')

    return validationLoss
    # TODO: add command for Tensorboard here


# %% Class to define a custom loss function for training, validation and testing - 01-06-2024
# NOTE: Function EvalLossFcn must be implemented using Torch operations to work!

class CustomLossFcn(nn.Module):
    # Class constructor
    def __init__(self, EvalLossFcn:callable) -> None:
        super(CustomLossFcn, self).__init__() # Call constructor of nn.Module
        if len((inspect.signature(EvalLossFcn)).parameters) >= 2:
            self.LossFcnObj = EvalLossFcn
        else: 
            raise ValueError('Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')    

    # Forward Pass evaluation method using defined EvalLossFcn
    def forward(self, predictVector, labelVector, params=None):
        lossBatch = self.LossFcnObj(predictVector, labelVector, params)
        return lossBatch.mean()
   
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
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

    L2regLoss = torch.norm(predictCorrection)**2 # Weighting the norm of the correction to keep it as small as possible

    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss
    return lossValue


# %% Function to save model state - 04-05-2024
def SaveModelState(model:nn.Module, modelName:str="trainedModel") -> None:
    if 'os.path' not in sys.modules:
        import os.path

    if modelName == 'trainedModel': 
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
        filename = "testModels/" + modelName
    else:
        filename = modelName
    
    # Attach timetag to model checkpoint
    #currentTime = datetime.datetime.now()
    #formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    #filename =  filename + "_" + formattedTimestamp
    print("Saving PyTorch Model State to:", filename)
    torch.save(model.state_dict(), filename) # Save model as internal torch representation

# %% Function to load model state - 04-05-2024 
def LoadModelState(model:nn.Module, modelName:str="trainedModel", filepath:str="testModels/") -> nn.Module:
    # Contatenate file path
    modelPath = os.path.join(filepath, modelName) #  + ".pth"
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
    
    # Function to validate path (check it is not completely black or white)
    def IsPatchValid(patchFlatten, lowerIntensityThr=5):
        
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
    

# %% TENSORBOARD functions - 04-06-2024
# Function to check if Tensorboard is running
def IsTensorboardRunning() -> bool:
    """Check if TensorBoard is already running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'tensorboard' in proc.info['cmdline']:
            #return proc.info['pid']
            return True
    return False

# Function to start TensorBoard process
def StartTensorboard(logDir:str) -> None:
    if not(IsTensorboardRunning):
        try:
            subprocess.Popen(['tensorboard', '--logdir', logDir, '--host', '0.0.0.0', '--port', '6006'])
            print('Tensorboard session successfully started using logDir:', logDir)
        except:
            RuntimeWarning('Tensorboard start-up failed. Continuing without opening session.')
    else:
        print('Tensorboard seems to be running in this session! Restarting with new directory...')
        #kill_tensorboard()
        #subprocess.Popen(['tensorboard', '--logdir', logDir, '--host', '0.0.0.0', '--port', '6006'])
        #print('Tensorboard session successfully started using logDir:', logDir)

# Function to stop TensorBoard process
def kill_tensorboard():
    """Kill all running TensorBoard instances."""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'tensorboard' in process.info['name']:
            for cmd in process.info['cmdline']:
                if 'tensorboard' in cmd:
                    print(f"Killing process {process.info['pid']}: {process.info['cmdline']}")
                    os.kill(process.info['pid'], signal.SIGTERM)

# Function to initialize Tensorboard session and writer
def ConfigTensorboardSession(logDir:str='./tensorboardLogs') -> SummaryWriter:

    print('Tensorboard logging directory:', logDir)
    StartTensorboard(logDir) 
    # Define writer # By default, this will write in a folder names "runs" in the directory of the main script. Else change providing path as first input.
    tensorBoardWriter = SummaryWriter(log_dir=logDir, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='') 

    # Return initialized writer
    return tensorBoardWriter


# %% Function to get model checkpoint and load it into nn.Module for training restart - 09-06-2024
def LoadModelAtCheckpoint(model:nn.Module, modelSavePath:str='./checkpoints', modelName:str='trainedModel', modelEpoch:int=0) -> nn.Module: 
    # TODO: add checks that model and checkpoint matches: how to? Check number of parameters?

    # Create path to model state file
    checkPointPath = os.path.join(modelSavePath, modelName + '_' + AddZerosPadding(modelEpoch, stringLength=4))

    # Attempt to load the model state and evaluate it
    if os.path.isfile(checkPointPath):
        print('Loading model to RESTART training from checkpoint: ', checkPointPath)
        try:
            loadedModel = LoadModelState(model, modelName, modelSavePath)
        except Exception as exception:
            print('Loading of model for training restart failed with error:', exception)
            print('Skipping reload and training from scratch...')
            return model
    else:
        raise ValueError('Specified model state file not found. Check input path.') 
    
    # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
    #with os.scandir(modelSavePath) as it:
        #modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        #modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]

    return loadedModel
    

# %% TRAINING and VALIDATION template function - 04-06-2024
def TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                              'device': GetDevice(), 
                                                                                                              'epochs': 10, 
                                                                                                              'Tensorboard':True,
                                                                                                              'saveCheckpoints':True,
                                                                                                              'checkpointsOutDir': './checkpoints',      
                                                                                                              'modelName': 'trainedModel',
                                                                                                              'loadCheckpoint': False,
                                                                                                              'lossLogName': 'Loss-value',
                                                                                                              'epochStart': 0}):
    # NOTE: is the default dictionary considered as "single" object or does python perform a merge of the fields?

    # Setup options from input dictionary
    taskType          = options['taskType']
    device            = options['device']
    numOfEpochs       = options['epochs']
    enableTensorBoard = options['Tensorboard']
    enableSave        = options['saveCheckpoints']
    checkpointDir     = options['checkpointsOutDir']
    modelName         = options['modelName']
    lossLogName       = options['lossLogName']
    epochStart        = options['epochStart']

    # Get Torch dataloaders
    if ('TrainingDataLoader' in dataloaderIndex.keys() and 'ValidationDataLoader' in dataloaderIndex.keys()):
        trainingDataset   = dataloaderIndex['TrainingDataLoader']
        validationDataset = dataloaderIndex['ValidationDataLoader']

        if not(isinstance(trainingDataset, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader". Check configuration.')
        if not(isinstance(validationDataset, DataLoader)):
            raise TypeError('Validation dataloader is not of type "DataLoader". Check configuration.')
            
    else:
        raise IndexError('Configuration error: either TrainingDataLoader or ValidationDataLoader is not a key of dataloaderIndex')

    # Configure Tensorboard
    if 'logDirectory' in options.keys():
        logDirectory = options['logDirectory']
    else:
        currentTime = datetime.datetime.now()
        formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute
        logDirectory = './tensorboardLog_' + modelName + formattedTimestamp
        
    if not(os.path.isdir(logDirectory)):
        os.mkdir(logDirectory)
    tensorBoardWriter = ConfigTensorboardSession(logDirectory)

    # If training is being restarted, attempt to load model
    if options['loadCheckpoint'] == True:
        model = LoadModelAtCheckpoint(model, options['checkpointsInDir'], modelName, epochStart)

    # Move model to device if possible (check memory)
    try:
        print('Moving model to selected device:', device)
        model = model.to(device) # Create instance of model using device 
    except Exception as exception:
        # Add check on error and error handling if memory insufficient for training on GPU:
        print('Attempt to load model in', device, 'failed due to error: ', repr(exception))


    # Training and validation loop
    input('\n-------- PRESS ENTER TO START TRAINING LOOP --------\n')
    trainLossHistory = np.zeros(numOfEpochs)
    validationLossHistory = np.zeros(numOfEpochs)

    for epochID in range(numOfEpochs):

        print(f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs}\n-------------------------------")
        # Do training over all batches
        trainLossHistory[epochID] = TrainModel(trainingDataset, model, lossFcn, optimizer, device, taskType) 
        # Do validation over all batches
        validationLossHistory[epochID] = ValidateModel(validationDataset, model, lossFcn, device, taskType) 

        # Update Tensorboard if enabled
        if enableTensorBoard:       
            #tensorBoardWriter.add_scalar(lossLogName + "/train", trainLossHistory[epochID], epochID + epochStart)
            #tensorBoardWriter.add_scalar(lossLogName + "/validation", validationLossHistory[epochID], epochID + epochStart)
            entriesTagDict = {'Training': trainLossHistory[epochID], 'Validation': validationLossHistory[epochID]}
            tensorBoardWriter.add_scalars(lossLogName, entriesTagDict, epochID)
            tensorBoardWriter.flush() 
        
        if enableSave:
            if not(os.path.isdir(checkpointDir)):
                os.mkdir(checkpointDir)

            modelSaveName = os.path.join(checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            SaveModelState(model, modelSaveName)
        
        # %% MODEL PREDICTION EXAMPLES
        examplePrediction, exampleLosses, inputSampleList = EvaluateModel(validationDataset, model, lossFcn, device, 10)

        # Add model graph using samples from EvaluateModel
        #if enableTensorBoard:       
            #tensorBoardWriter.add_graph(model, inputSampleList, verbose=False)
            #tensorBoardWriter.flush() 

        print('\n  Random Sample predictions from validation dataset:\n')
        torch.set_printoptions(precision=2)
        for id in range(examplePrediction.shape[0]):
            print('\tPrediction: ', examplePrediction[id, :].tolist(), ' --> Loss: ',exampleLosses[id].tolist())
        
        torch.set_printoptions(precision=5)

    return model, trainLossHistory, validationLossHistory, inputSampleList

# %% Model evaluation function on a random number of samples from dataset - 06-06-2024
def EvaluateModel(dataloader:DataLoader, model:nn.Module, lossFcn: nn.Module, device=GetDevice(), numOfSamples:int=10) -> np.array:
        
    model.eval()
    with torch.no_grad(): 

        # Get some random samples from dataloader as list
        extractedSamples = GetSamplesFromDataset(dataloader, numOfSamples)

        # Create input array as torch tensor
        X = torch.zeros(len(extractedSamples), extractedSamples[0][0].shape[0])
        Y = torch.zeros(len(extractedSamples), extractedSamples[0][1].shape[0])

        for id, (inputVal, labelVal) in enumerate(extractedSamples):
            X[id, :] = inputVal
            Y[id, :] = labelVal

        # Perform FORWARD PASS
        examplePrediction = model(X.to(device)) # Evaluate model at input

        # Compute loss for each input separately
        exampleLosses = torch.zeros(examplePrediction.size(0))

        inputSampleList = []
        for id in range(examplePrediction.size(0)):
            
            # Get prediction and label samples 
            inputSampleList.append(examplePrediction[id, :].reshape(1, -1))
            labelSample = Y[id,:].reshape(1, -1)

            # Evaluate
            exampleLosses[id] = lossFcn(inputSampleList[id].to(device), labelSample.to(device)).item()

    return examplePrediction, exampleLosses, inputSampleList
                
# %% Function to extract specified number of samples from dataloader - 06-06-2024
def GetSamplesFromDataset(dataloader: DataLoader, numOfSamples:int=10):

    samples = []
    for batch in dataloader:
        for sample in zip(*batch): # Construct tuple (X,Y) from batch
            samples.append(sample)

            if len(samples) == numOfSamples:
                return samples
                   
    return samples




# %% Torch to/from ONNx format exporter/loader based on TorchDynamo (PyTorch >2.0) - 09-06-2024
def ExportTorchModelToONNx(model:nn.Module, dummyInputSample:torch.tensor, onnxExportPath:str='.', onnxSaveName:str='trainedModelONNx', modelID:int=0) -> None:

    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else: 
        stringLength = 3

    modelSaveName = os.path.join(onnxExportPath, onnxSaveName + AddZerosPadding(modelID, stringLength))

    # Export model to ONNx object
    modelONNx = torch.onnx.dynamo_export(model, dummyInputSample) # NOTE: ONNx model is stored as a binary protobuf file!
    # Save ONNx model 
    modelONNx.save(modelSaveName)

def LoadTorchModelFromONNx(dummyInputSample:torch.tensor, onnxExportPath:str='.', onnxSaveName:str='trainedModelONNx', modelID:int=0):
    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else: 
        stringLength = 3

    modelSaveName = os.path.join(onnxExportPath, onnxSaveName + AddZerosPadding(modelID, stringLength))

    if os.path.isfile():
            modelONNx = onnx.load(modelSaveName)
            torchModel = None
            return torchModel, modelONNx
    else: 
        raise ImportError('Specified input path to .onnx model not found.')

# %% Model Graph visualization function based on Netron module #TODO



# %% Other auxiliary functions - 09-06-2024
def AddZerosPadding(intNum:int, stringLength:str=4):
    return f"{intNum:0{stringLength}d}" # Return strings like 00010

# %% MATLAB wrapper class for Torch models evaluation - TODO DD-06-2024
#class TorchModel_MATLABwrap():
#
#    def __init__(self) -> None:
#        print('TODO: Model loader and setup')
#    
#    def forward():
#        print('TODO: model evaluation method')
#
# %% TORCH to ONNX format model converter - TODO




# %% MAIN 
def main():
    print('In this script, main does actually nothing ^_^.')
    
if __name__== '__main__':
    main()