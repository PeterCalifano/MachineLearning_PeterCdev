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
from onnx import version_converter
from typing import Union

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

    batchValueForPrint = np.floor(len(dataloader)/100)

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

        if batchCounter % batchValueForPrint == 0: # Print loss value 
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
    '''Custom loss function based class, instantiated by specifiying a loss function (callable object) and optionally, a dictionary containing parameters required for the evaluation'''
    def __init__(self, EvalLossFcn:callable, paramsTrain:dict = None, paramsEval:dict = None) -> None:
        '''Constructor for CustomLossFcn class'''
        super(CustomLossFcn, self).__init__() # Call constructor of nn.Module
        if len((inspect.signature(EvalLossFcn)).parameters) >= 2:
            self.LossFcnObj = EvalLossFcn
        else: 
            raise ValueError('Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')    

        # Store loss function parameters dictionary
        self.paramsTrain = paramsTrain 

        if paramsEval == None:
            # Assign training parameters to evaluation parameters if not specified
            self.paramsEval = self.paramsTrain 

    def forward(self, predictVector, labelVector):
        ''''Forward pass method to evaluate loss function on input and label vectors using EvalLossFcn'''
        lossBatch = self.LossFcnObj(predictVector, labelVector, self.paramsTrain, self.paramsEval)
        return lossBatch.mean()
   

# %% Function to save model state - 04-05-2024, updated 11-06-2024
def SaveTorchModel(model:nn.Module, modelName:str="trainedModel", saveAsTraced:bool=False, exampleInput=None, targetDevice:str='cpu') -> None:
    if 'os.path' not in sys.modules:
        import os.path

    if saveAsTraced: 
        extension = '.pt'
    else:
        extension = '.pth'
        
    # Format target device string to remove ':' from name
    targetDeviceName = targetDevice
    targetDeviceName = targetDeviceName.replace(':', '') 

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

        filename = "testModels/" + modelName + '_' + targetDeviceName + extension 
    else:
        filename = modelName  + '_' + targetDeviceName + extension 
    
    # Attach timetag to model checkpoint
    #currentTime = datetime.datetime.now()
    #formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    #filename =  filename + "_" + formattedTimestamp
    print("Saving PyTorch Model State to:", filename)

    if saveAsTraced:
        print('Saving traced model...')
        if exampleInput is not None:
            tracedModel = torch.jit.trace((model).to(targetDevice), exampleInput.to(targetDevice))
            tracedModel.save(filename)
            print('Model correctly saved with filename: ', filename)
        else: 
            raise ValueError('You must provide an example input to trace the model through torch.jit.trace()')
    else:
        print('Saving NOT traced model...')
        torch.save(model.state_dict(), filename) # Save model as internal torch representation
        print('Model correctly saved with filename: ', filename)


# %% Function to load model state into empty model- 04-05-2024, updated 11-06-2024
def LoadTorchModel(model:nn.Module=None, modelName:str="trainedModel", filepath:str="testModels/", loadAsTraced:bool=False) -> nn.Module:
    
    # Check if input name has extension
    modelNameCheck, extension = os.path.splitext(str(modelName))

    #print(modelName, ' ', modelNameCheck, ' ', extension)

    if extension != '.pt' and extension != '.pth':
        if loadAsTraced: 
            extension = '.pt'
        else:
            extension = '.pth'
    else:
        extension = ''      

    # Contatenate file path
    modelPath = os.path.join(filepath, modelName + extension) 

    if not(os.path.isfile(modelPath)):
        raise FileNotFoundError('Model specified by: ', modelPath, ': NOT FOUND.')
    
    if loadAsTraced and model is None:
        print('Loading traced model from filename: ', modelPath)
        # Load traced model using torch.jit
        model = torch.jit.load(modelPath)
        print('Traced model correctly loaded.')

    elif not(loadAsTraced) or (loadAsTraced and model is not None):

        if loadAsTraced and model is not None:
            print('loadAsTraced is specified as true, but model has been provided. Loading from state: ', modelPath)
        else: 
            print('Loading model from filename: ', modelPath)

        # Load model from file
        model.load_state_dict(torch.load(modelPath))
        # Evaluate model to set (weights, biases)
        model.eval()

    else:
        raise ValueError('Incorrect combination of inputs! Valid options: \n  1) model is None AND loadAsTraced is True; \n  2) model is nn.Module AND loadAsTraced is False; \n  3) model is nn.Module AND loadAsTraced is True (fallback to case 2)')

    return model


# %% Function to save Dataset object - 01-06-2024
def SaveTorchDataset(datasetObj:Dataset, datasetFilePath:str='', datasetName:str='dataset') -> None:

    if not(os.path.isdir(datasetFilePath)):
        os.makedirs(datasetFilePath)
    torch.save(datasetObj, datasetFilePath + datasetName + ".pt")

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
        except Exception as errMsg:
            print('Failed due to:', errMsg, '. Continuing without opening session.')
    else:
        print('Tensorboard seems to be running in this session! Restarting with new directory...')
        #kill_tensorboard()
        #subprocess.Popen(['tensorboard', '--logdir', logDir, '--host', '0.0.0.0', '--port', '6006'])
        #print('Tensorboard session successfully started using logDir:', logDir)

# Function to stop TensorBoard process
def KillTensorboard():
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
            loadedModel = LoadTorchModel(model, modelName, modelSavePath)
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
    #input('\n-------- PRESS ENTER TO START TRAINING LOOP --------\n')
    trainLossHistory = np.zeros(numOfEpochs)
    validationLossHistory = np.zeros(numOfEpochs)

    for epochID in range(numOfEpochs):

        print(f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
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

            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(1, -1) # Get single input sample for model saving
            modelSaveName = os.path.join(checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            SaveTorchModel(model, modelSaveName, saveAsTraced=True, exampleInput=exampleInput, targetDevice=device)
        
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

def EvaluateModel(dataloader:DataLoader, model:nn.Module, lossFcn: nn.Module, device=GetDevice(), numOfSamples:int=10, inputSample:torch.tensor=None) -> np.array:
    '''Torch model evaluation function to perform inference using either specified input samples or input dataloader'''
    model.eval() # Set model in prediction mode
    with torch.no_grad(): 
        if inputSample is None:
            # Get some random samples from dataloader as list
            extractedSamples = GetSamplesFromDataset(dataloader, numOfSamples)

            # Create input array as torch tensor
            X = torch.zeros(len(extractedSamples), extractedSamples[0][0].shape[0])
            Y = torch.zeros(len(extractedSamples), extractedSamples[0][1].shape[0])

            #inputSampleList = []
            for id, (inputVal, labelVal) in enumerate(extractedSamples):
                X[id, :] = inputVal
                Y[id, :] = labelVal

            #inputSampleList.append(inputVal.reshape(1, -1))

            # Perform FORWARD PASS
            examplePredictions = model(X.to(device)) # Evaluate model at input

            # Compute loss for each input separately
            exampleLosses = torch.zeros(examplePredictions.size(0))

            examplePredictionList = []
            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples 
                examplePredictionList.append(examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id,:].reshape(1, -1)

                # Evaluate loss function
                exampleLosses[id] = lossFcn(examplePredictionList[id].to(device), labelSample.to(device)).item()
   
        else:
            # Perform FORWARD PASS # NOTE: NOT TESTED
            X = inputSample
            examplePredictions = model(X.to(device)) # Evaluate model at input

            examplePredictionList = []
            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples 
                examplePredictionList.append(examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id,:].reshape(1, -1)

                # Evaluate loss function
                exampleLosses[id] = lossFcn(examplePredictionList[id].to(device), labelSample.to(device)).item()

        return examplePredictions, exampleLosses, X.to(device)

                
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
def ExportTorchModelToONNx(model:nn.Module, dummyInputSample:torch.tensor, onnxExportPath:str='.', onnxSaveName:str='trainedModelONNx', modelID:int=0, onnx_version = None):

    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else: 
        stringLength = 3

    modelSaveName = os.path.join(onnxExportPath, onnxSaveName + AddZerosPadding(modelID, stringLength))

    # Export model to ONNx object
    modelONNx = torch.onnx.dynamo_export(model, dummyInputSample) # NOTE: ONNx model is stored as a binary protobuf file!
    #modelONNx = torch.onnx.export(model, dummyInputSample) # NOTE: ONNx model is stored as a binary protobuf file!

    # Save ONNx model 
    pathToModel = modelSaveName+'.onnx'
    modelONNx.save(pathToModel) # NOTE: this is a torch utility, not onnx!

    # Try to convert model to required version
    if (onnx_version is not None) and type(onnx_version) is int:
        convertedModel=None
        print('Attempting conversion of ONNx model to version:', onnx_version)
        try:
            print(f"Model before conversion:\n{modelONNx}")
            # Reload onnx object using onnx module
            tmpModel = onnx.load(pathToModel)
            # Convert model to get new model proto
            convertedModelProto = version_converter.convert_version(tmpModel, onnx_version)

            # TEST
            #convertedModelProto.ir_version = 7

            # Save model proto to .onnbx
            onnx.save_model(convertedModelProto, modelSaveName + '_ver' + str(onnx_version) + '.onnx')

        except Exception as errorMsg:
            print('Conversion failed due to error:', errorMsg)
    else: 
        convertedModel=None

    return modelONNx, convertedModel


def LoadTorchModelFromONNx(dummyInputSample:torch.tensor, onnxExportPath:str='.', onnxSaveName:str='trainedModelONNx', modelID:int=0):
    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else: 
        stringLength = 3

    modelSaveName = os.path.join(onnxExportPath, onnxSaveName + '_', AddZerosPadding(modelID, stringLength))

    if os.path.isfile():
            modelONNx = onnx.load(modelSaveName)
            torchModel = None
            return torchModel, modelONNx
    else: 
        raise ImportError('Specified input path to .onnx model not found.')

# %% Model Graph visualization function based on Netron module # TODO



# %% Other auxiliary functions - 09-06-2024
def AddZerosPadding(intNum:int, stringLength:str=4):
    return f"{intNum:0{stringLength}d}" # Return strings like 00010


#inputImageSize:list, kernelSizes:list, OutputChannelsSizes:list, PoolingLayersSizes:list, inputChannelSize:int=1, withBiases=True

# Auxiliar functions
def ComputeConv2dOutputSize(inputSize:Union[list, np.array, torch.tensor], kernelSize=3, strideSize=1, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D convolutional layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
    return int((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1), int((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1)

def ComputePooling2dOutputSize(inputSize:Union[list, np.array, torch.tensor], kernelSize=2, strideSize=2, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D max/avg pooling layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
    return int(( (inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1), int(( (inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1)

# ConvBlock 2D and flatten sizes computation (SINGLE BLOCK)
def ComputeConvBlockOutputSize(inputSize:Union[list, np.array, torch.tensor], outChannelsSize:int, 
                               convKernelSize:int=3, poolingkernelSize:int=2, 
                               convStrideSize:int=1, poolingStrideSize:int=1, 
                               convPaddingSize:int=0, poolingPaddingSize:int=0):
    
    # TODO: modify interface to use something like a dictionary with the parameters, to make it more fexible and avoid the need to pass all the parameters

    '''Compute output size and number of features maps (channels, i.e. volume) of a ConvBlock layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''

    # Compute output size of Conv2d and Pooling2d layers
    conv2dOutputSize = ComputeConv2dOutputSize(inputSize, convKernelSize, convStrideSize, convPaddingSize)
    convBlockOutputSize = ComputePooling2dOutputSize(conv2dOutputSize, poolingkernelSize, poolingStrideSize, poolingPaddingSize)

    # Compute total number of features after ConvBlock as required for the fully connected layers
    conv2dFlattenOutputSize = convBlockOutputSize[0] * convBlockOutputSize[1] * outChannelsSize

    return convBlockOutputSize, conv2dFlattenOutputSize


# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024
class TorchModel_MATLABwrap():
    def __init__(self, trainedModelName:str, trainedModelPath:str) -> None:
        # Get available device
        self.device = GetDevice()

        # Load model state and state
        trainedModel = LoadTorchModel(None, trainedModelName, trainedModelPath, loadAsTraced=True)

        self.trainedModel = trainedModel.to(self.device)


    def forward(self, inputSample:np.array, inputSize:int=None):
        '''Forward method to perform inference for ONE sample input using trainedModel'''
        if inputSample.dtype is not np.float32:
            inputSample = np.float32(inputSample)

        # TODO: check the input is exactly identical to what the model receives using EvaluateModel() loading from dataset!        
        # Convert numpy array into torch.tensor for model inference
        if inputSize == None:
            X = torch.tensor(inputSample).reshape(1, -1)
        else:
            # Compute number of batches
            inputVecLen = inputSample.size()
            numBatches = inputVecLen / inputSize

            # Check if numBatches is an integer
            if not(numBatches.is_integer()):
                raise ValueError('Specified input size causes number of batches to be fractional. Check input sample size.')
            
            X = torch.tensor(inputSample).reshape(numBatches, -1)

        # ########### DEBUG ######################: 
        print('Evaluating model using batch input: ', X)
        ############################################

        # Perform inference using model
        Y = self.trainedModel(X.to(self.device))

        # ########### DEBUG ######################: 
        print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy() # Move to cpu and convert to numpy
    
        

# %% Training and validation manager class - 22-06-2024 (WIP)
# TODO: Features to include: 
# 1) Multi-process/multi-threading support for training and validation of multiple models in parallel
# 2) Logging of all relevat options and results to file (either csv or text from std output)
# 3) Main training logbook to store all data to be used for model selection and hyperparameter tuning, this should be "per project"

class TrainAndValidationManager():
    '''Class to manage training and validation of PyTorch models using specified datasets and loss functions.'''

    def __init__(self, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                     'device': GetDevice(), 
                                                                                     'epochs': 10, 
                                                                                     'Tensorboard':True,
                                                                                     'saveCheckpoints':True,
                                                                                     'checkpointsOutDir': './checkpoints',      
                                                                                     'modelName': 'trainedModel',
                                                                                     'loadCheckpoint': False,
                                                                                     'lossLogName': 'Loss-value',
                                                                                     'epochStart': 0}):
        
        '''Constructor for TrainAndValidationManager class. Initializes model, loss function, optimizer and training/validation options.'''

        # Define manager parameters
        self.model = model
        self.lossFcn = lossFcn

        # Optimizer --> # TODO: check how to modify learning rate and momentum while training
        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        elif isinstance(optimizer, int):
                if optimizer == 0:
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=learnRate, momentum=momentumValue) 
                elif optimizer == 1:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=learnRate)
                else:
                    raise ValueError('Optimizer type not recognized. Use either 0 for SGD or 1 for Adam.')
        else:
            raise ValueError('Optimizer must be either an instance of torch.optim.Optimizer or an integer representing the optimizer type.')
        
        # Define training and validation options
    
    def LoadDatasets(self, dataloaderIndex:dict):
        '''Method to load datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)'''
        # TODO: Load all datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)
        pass

    def TrainAndValidateModel(self):
        '''Method to train and validate model using loaded datasets and specified options'''
        pass

    def GetTracedModel(self):
        pass
    


# %% MAIN 
def main():
    print('In this script, main does actually nothing ^_^.')
    
if __name__== '__main__':
    main()