# Script created by PeterC 17-05-2024 reproducing model presentend in Pugliatti's PhD thesis (C4, Table 4.1)
# Implemented in PyTorch. Datasets on Zenodo: https://zenodo.org/records/7107409

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime