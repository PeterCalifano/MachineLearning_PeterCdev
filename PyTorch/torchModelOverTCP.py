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
import tcpServerPy



# MAIN SCRIPT
def main():
    print('\n\n----------------------------------- RUNNING: torchModelOverTCP.py -----------------------------------\n')
    print("MAIN script operations: initialize always-on server --> listen to data from client --> if OK, evaluate model --> if OK, return output to client\n")
    
    # %% TORCH MODEL LOADING
    # Model path
    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev'
    modelID = 5
    tracedModelName = 'HorizonPixCorrector_CNNv2_' + customTorch.AddZerosPadding(modelID, 3) + '_cpu'
    tracedModelName = 'HorizonPixCorrector_CNNv2_005_cpu' + '.pt'

    # Parameters

    # Load torch traced model from file
    torchWrapper = customTorch.TorchModel_MATLABwrap(tracedModelName, tracedModelSavePath)

    # %% TCP SERVER INITIALIZATION
    HOST, PORT = "127.0.0.1", 50000 # Define host and port (random is ok)

    # Define DataProcessor object for RequestHandler
    numOfBytes = 56*4 # Length of input * number of bytes in double
    dataProcessorObj = tcpServerPy.DataProcessor(torchWrapper.forward, np.float32, numOfBytes)

    # Initialize TCP server and keep it running
    with tcpServerPy.pytcp_server((HOST, PORT), tcpServerPy.pytcp_requestHandler, dataProcessorObj, bindAndActivate=True) as server:
        try:
            print('\nServer initialized correctly. Set in "serve_forever" mode.')
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer is gracefully shutting down =D.")
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    main()