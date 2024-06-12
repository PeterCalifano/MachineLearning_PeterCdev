# Script created by PeterC 05-06-2024 to analyze performance of CNN-NN network enhancing pixel extraction for Limb based Navigation

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))

import customTorch # Custom torch tools
import limbPixelExtraction_CNN_NN
import datasetPreparation

import torch
import datetime
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import datetime
import numpy as np



def main():
    print('TODO')


if __name__ == '__main__':
    main()