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
import sys
import subprocess
import psutil

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
    print(f"Using {device} device")
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

            elif taskType.lower() == 'regression':
                print('TODO')

            elif taskType.lower() == 'custom':
                print('TODO')


    if taskType.lower() == 'classification': 
        validationLoss /= numberOfBatches # Compute batch size normalized loss value
        correctOuputs /= size # Compute percentage of correct classifications over batch size
        print(f"Validation (Classification): \n Accuracy: {(100*correctOuputs):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'regression':
        print('TODO')
        print(f"Validation (Regression): \n Avg absolute accuracy: {avgAbsAccuracy:>0.1f}, Avg relative accuracy: {(100*avgRelAccuracy):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'custom':
        print('TODO')

    return validationLoss
    # TODO: add command for Tensorboard here


# %% Class to define a custom loss function for training, validation and testing - 01-06-2024
# NOTE: Function EvalLossFcn must be implemented using Torch operations to work!

class CustomLossFcn(nn.Module):
    # Class constructor
    def __init__(self, EvalLossFcn:function) -> None:
        super(CustomLossFcn, self).__init__() # Call constructor of nn.Module
        if len(EvalLossFcn) >= 2:
            self.LossFcnObj = EvalLossFcn
        else: 
            raise ValueError('Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')    

    # Forward Pass evaluation method using defined EvalLossFcn
    def forward(self, predictVector, labelVector, params=None):
        lossBatch = self.LossFcnObj(predictVector, labelVector, params)
        return lossBatch.mean()
   
# %% Custom loss function for Moon Limb pixel extraction CNN enhancer - 01-06-2024
def MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, params:list=None):
    # alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    coeff = 0.5
    LimbConicMatrixImg = np.reshape(labelVector, 3, 3)
    # Evaluate loss
    conicLoss = torch.matmul(predictCorrection.T, torch.matmul(LimbConicMatrixImg, predictCorrection)) # Weighting violation of Horizon conic equation
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
    currentTime = datetime.datetime.now()
    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    filename =  filename + "_" + formattedTimestamp
    print("Saving PyTorch Model State to:", filename)
    torch.save(model.state_dict(), filename) # Save model as internal torch representation

# %% Function to load model state - 04-05-2024 
def LoadModelState(model:nn.Module, modelName:str="trainedModel", filepath:str="testModels/") -> nn.Module:
    # Contatenate file path
    modelPath = filepath + modelName + ".pth"
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
# First prototype completed by PC - 04-06-2024
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
            print('Tensorboard session successfully started.')
        except:
            RuntimeWarning('Tensorboard start-up failed. Continuing without opening session.')
    else:
        print('Tensorboard seems to be running in this session!')

# Function to initialize Tensorboard session and writer
def ConfigTensorboardSession(logDir:str='./tensorboardLogs') -> SummaryWriter:

    print('Tensorboard logging directory:', logDir)
    StartTensorboard(logDir) 
    # Define writer # By default, this will write in a folder names "runs" in the directory of the main script. Else change providing path as first input.
    tensorBoardWriter = SummaryWriter(log_dir=logDir, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='') 

    # Return initialized writer
    return tensorBoardWriter

# %% TRAINING and VALIDATION template function
def TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                              'device': GetDevice(), 
                                                                                                              'epochs': 10, 
                                                                                                              'Tensorboard':True,
                                                                                                              'saveCheckpoints':True,
                                                                                                              'checkpointsDir:': './checkpoints',
                                                                                                              'modelName': 'trainedModel',
                                                                                                              'loadCheckpoint': False}):

    # Setup options from input dictionary
    taskType          = options['classification']
    device            = options['device']
    numOfEpochs       = options['epochs']
    enableTensorBoard = options['Tensorboard']
    enableSave        = options['saveCheckpoints']
    checkpointDir     = options['checkpointsDir']
    modelName         = options['modelName']
    
    ##### DEVNOTE #####
    if options['loadCheckpoint'] == True:
        raise NotImplementedError('Current version does not support loading model checkpoints yet!')
    ###################

    # Get Torch dataloaders
    if ('TrainingDataLoader' in dataloaderIndex.keys() and 'ValidationDataLoader' in dataloaderIndex.keys()):
        trainingDataset   = dataloaderIndex['TrainingDataLoader']
        validationDataset = dataloaderIndex['ValidationDataLoader']

        if not(trainingDataset is DataLoader):
            raise TypeError('Training dataloader is not of type "DataLoader". Check configuration.')
        if not(validationDataset is DataLoader):
            raise TypeError('Validation dataloader is not of type "DataLoader". Check configuration.')
            
    else:
        raise IndexError('Configuration error: either TrainingDataLoader or ValidationDataLoader is not a key of dataloaderIndex')

    # Configure Tensorboard
    if 'logDirectory' in options.keys():
        logDirectory = options['logDirectory']
        tensorBoardWriter = ConfigTensorboardSession(logDirectory)
    else:
        tensorBoardWriter = ConfigTensorboardSession()

    # Move model to device if possible (check memory)
    try:
        print('Moving model to selected device:', device)
        model = model.to(device) # Create instance of model using device 
    except Exception as exception:
        # Add check on error and error handling if memory insufficient for training on GPU:
        print('Attempt to load model in', device, 'failed due to error: ', repr(exception))


    # Training and validation loop
    input('-------- PRESS ENTER TO START TRAINING LOOP --------')
    trainLossHistory = []
    validationLossHistory = []

    for epochID in range(numOfEpochs):

        print(f"Epoch {epochID}\n-------------------------------")
        # Do training over all batches
        trainLossHistory[epochID] = TrainModel(trainingDataset, model, lossFcn, optimizer, device, taskType) 

        # Do validation over all batches
        validationLossHistory[epochID] = ValidateModel(validationDataset, lossFcn, device, taskType) 

        # Update Tensorboard if enabled
        if enableTensorBoard:       
            tensorBoardWriter.add_scalar("Loss/train", trainLossHistory[epochID], epochID)
            tensorBoardWriter.add_scalar("Loss/validation", validationLossHistory[epochID], epochID)
            tensorBoardWriter.flush() 
        
        if enableSave:
            modelSaveName = checkpointDir + modelName + '_' + epochID
            SaveModelState(model, modelSaveName)
        

    return model, trainLossHistory, validationLossHistory

# %% MAIN 
def main():
    print('In this script, main does actually nothing ^_^.')
    
if __name__== '__main__':
    main()