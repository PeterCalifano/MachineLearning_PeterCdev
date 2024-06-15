"""! Prototype script for torch model instantiation and evaluation over TCP, created by PeterC - 15-06-2024"""

# Python imports
import torch
from torch import nn 
import sys, os

# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/tcpServerPy'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

# Custom imports
import customTorch
import limbPixelExtraction_CNN_NN
import tcpServerPy


# MAIN SCRIPT
def main():
    print('MAIN script operations: initialize always-on server --> listen to data from client --> if OK, evaluate model --> if OK, return output to client')
    
    # %% TORCH MODEL LOADING
    # Model path
    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev'
    tracedModelName = 'trainedModel_' + customTorch.AddZerosPadding(0, 4)

    # Parameters
    device = customTorch.GetDevice()

    # Load torch traced model from file
    trainedTracedModel = customTorch.LoadTorchModel(None, tracedModelName, tracedModelSavePath, True).to(device)

    # %% TCP SERVER INITIALIZATION



if __name__ == "__main__":
    main()