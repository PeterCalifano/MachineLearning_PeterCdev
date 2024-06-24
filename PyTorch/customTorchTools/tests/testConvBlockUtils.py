'''
Test Script created by PeterC 04-06-2024 to verify the correct implementation of the utility functions for the CNNs and NNs in the customTorch module.
'''

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import customTorchTools  # Custom torch tools

import torch
import datetime
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
from typing import Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim


def main():
    # Test definition of ConvBlock
    outChannelsSizes = [16, 32, 75, 15] 

    # Test computation of output size of Conv2d using default settings
    patchSize = [7, 7]
    convKernelSize = 3
    convStrideSize = 1
    convPaddingSize = 0
    conv2dOutputSize = customTorchTools.ComputeConv2dOutputSize(patchSize, convKernelSize, convStrideSize, convPaddingSize)

    # Test computation of output size of Pooling2d using default settings
    poolingkernelSize = 2
    poolingStrideSize = 1
    poolingOutputSize = customTorchTools.ComputePooling2dOutputSize([5,5], poolingkernelSize, poolingStrideSize)

    print('Output size of Conv2d:', conv2dOutputSize)
    print('Output size of Pooling2d:', poolingOutputSize)

    # Test computation of number of features after ConvBlock using default settings
    convBlockOutputSize = customTorchTools.ComputeConvBlockOutputSize([7,7], outChannelsSizes[0])

    print('Output size of ConvBlock:', convBlockOutputSize)



if __name__ == '__main__':
    main()