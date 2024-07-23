# Script implementing a classification example on CIFAR100 as exercise by PeterC, using pretrained models in torchvision - 23-07-2024
# Reference: https://pytorch.org/vision/0.9/models.html#classification

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import numpy as np

import sys, os
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
import customTorchTools
import torchvision.models as models

def main():
    model = models.resnet18(weights=None) 
    # NOTE: all models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    print(model)

    # Therefore, let's modify the last layer! (Named fc)
    numInputFeatures = model.fc.in_features # Same as what the model uses 
    numOutClasses = 10 # Selected by user 

    model.fc = nn.Linear(in_features=numInputFeatures,
                         out_features=numOutClasses, bias=True)  # 2 classes in our case
    
    print(model) # Check last layer now

    # Load CIFAR10 dataset
    trainData = datasets.CIFAR10(root='data',
                                 train=True,
                                 download=True,
                                 transform=ToTensor())


if __name__ == '__main__':
    main()