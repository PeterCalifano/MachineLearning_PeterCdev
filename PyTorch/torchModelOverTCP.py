"""! Prototype script for torch model instantiation and evaluation over TCP, created by PeterC - 15-06-2024"""

# Python imports
import torch
from torch import nn 
import sys, os

# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/tcpServerPy'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import numpy as np

# Custom imports
import customTorch
import limbPixelExtraction_CNN_NN
import tcpServerPy



# MAIN SCRIPT
def main():
    print('\n\n--------------------------- RUNNING: torchModelOverTCP.py ---------------------------n')
    print('MAIN script operations: initialize always-on server --> listen to data from client --> if OK, evaluate model --> if OK, return output to client\n\n')
    
    # %% TORCH MODEL LOADING
    # Model path
    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev'
    tracedModelName = 'trainedTracedModel' + customTorch.AddZerosPadding(75, 3)

    # Parameters

    # Load torch traced model from file
    torchWrapper = customTorch.TorchModel_MATLABwrap(tracedModelName, tracedModelSavePath)

    # %% TCP SERVER INITIALIZATION
    HOST, PORT = "localhost", 65333 # Define host and port (random is ok)

    # Define DataProcessor object for RequestHandler
    dataProcessorObj = tcpServerPy.DataProcessor(torchWrapper.forward, np.ndarray)

    # Initialize TCP server and keep it running
    with tcpServerPy.pytcp_server((HOST, PORT), tcpServerPy.pytcp_requestHandler, bindAndActivate=True) as server:
        print('Server initialized correctly. Set in "serve_forever" mode.')
        #server.serve_forever()

if __name__ == "__main__":
    main()