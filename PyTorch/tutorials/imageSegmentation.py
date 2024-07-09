# Script implementing a segmentation example as exercise by PeterC - DD-05-2024
# Reference: 

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime
import numpy as np