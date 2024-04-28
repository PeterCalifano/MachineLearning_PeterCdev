# Example from norse documentation: Spiking Convolutional Classifier
# Created by PeterC 28-04-2024

import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers

model = SequentialState(
    nn.Conv2d(1, 20, 5, 1),      # Convolve from 1 -> 20 channels
    LIFCell(),                   # Spiking activation layer
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),     # Convolve from 20 -> 50 channels
    LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),                # Flatten to 800 units
    nn.Linear(800, 10),
    LICell(),                    # Non-spiking integrator layer
)

# RANDOM DATA
data = torch.randn(8, 1, 28, 28) # 8 batches, 1 channel, 28x28 pixels
output, state = model(data)      # Provides a tuple (tensor (8, 10), neuron state)