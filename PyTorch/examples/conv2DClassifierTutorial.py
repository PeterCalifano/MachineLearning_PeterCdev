# Script implementing the PyTorch "Training a Classifier" example as exercise by PeterC - 04-05-2024
# Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
import torchvision.transforms as transforms # Import transform utilities from torchvision
import datetime

# %% PREPARE AND VISUALIZE DATASET 
