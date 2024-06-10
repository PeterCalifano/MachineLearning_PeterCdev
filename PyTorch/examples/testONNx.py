# Script created by PeterC to test ONNx related codes (mainly from Torch) - 10-06-2024
import onnx 
import torch
from torch import nn 

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))


import customTorch # Custom torch tools
import limbPixelExtraction_CNN_NN

import datetime
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

def main():

    # Define inputs
    exportPath = './ExportedModelsToONNx'
    modelSavePath = './checkpoints/HorizonPixCorrector_CNN_run8'
    modelName = 'trainedModel'
    datasetSavePath = modelSavePath + 'sampleDatasetToONNx'
    batch_size = 16

    # NOTE: these settings must be the same as the saved model. Current version does not check for this.
    outChannelsSizes = [16, 32, 75, 15]
    kernelSizes = [3, 1]

    modelEmpty = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNN(outChannelsSizes, kernelSizes)

    # Load torch model and define loss function
    trainedModel = customTorch.LoadModelState(modelEmpty, modelName, modelSavePath)

    lossFcn = customTorch.CustomLossFcn(customTorch.MoonLimbPixConvEnhancer_LossFcn)

    # Load sample dataset
    sampleData = customTorch.LoadTorchDataset(datasetSavePath)
    sampleDataset  = DataLoader(sampleData, batch_size, shuffle=True)

    # Test model and get sample inputs
    examplePrediction, exampleLosses, inputSampleList = customTorch.EvaluateModel(sampleDataset, trainedModel, lossFcn)

    # Convert to ONNx format and save
    modelONNx = customTorch.ExportTorchModelToONNx(trainedModel, inputSampleList[0], exportPath, modelName, 0)

if __name__ == '__main__':
    main()