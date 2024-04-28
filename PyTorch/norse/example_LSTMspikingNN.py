# Example from norse documentation: Long short-term spiking neural networks
# Taken from paper: https://arxiv.org/abs/1803.09574
# Created by PeterC 28-04-2024

import torch
from norse.torch import LSNNRecurrent
# Recurrent LSNN network with 2 input neurons and 10 output neurons
layer = LSNNRecurrent(2, 10)
# Generate data: 20 timesteps with 8 datapoints per batch for 2 neurons
data  = torch.zeros(20, 8, 2)
# Tuple of (output spikes of shape (20, 8, 2), layer state)
output, new_state = layer(data)