# Import modules
import sys, os

# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import customTorchTools # Custom torch tools
import limbPixelExtraction_CNN_NN  # Custom model classes
import datasetPreparation
from sklearn import preprocessing # Import scikit-learn for dataset preparation

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


def ComputeRandomDisplacements():
    


def main():
    
    print('------------------------------- TEST: Loss functions classes -------------------------------')

    LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch, 
                  # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
    # Loss function parameters
    params = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 0}

    lossFcn = customTorchTools.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor, params)
    #model = ModelClasses.HorizonExtractionEnhancerCNNv3maxDeeper

    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev/checkpoints/HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run0003'
    tracedModelName = 'HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run0003_0002_cuda0.pt'


    # Load torch traced model from file
    torchWrapper = customTorchTools.TorchModel_MATLABwrap(tracedModelName, tracedModelSavePath)

    modelParams = torchWrapper.trainedModel.parameters()




    

if __name__ == '__main__':
    main()