'''
Script to test the Polar-n-direction distance function for the evaluation of the distance of a point from a conic in the image place represented by its [3x3] matrix.
Created by PeterC, 26-06-2024. Reference: “[14] Y. Wu, H. Wang, F. Tang, and Z. Wang, “Efficient conic fitting with an analytical polar-n-direction geometric distance,
” Pattern Recognition, vol. 90, pp. 415-423, 2019.” 
'''

import sys
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

# Import the required modules
import torch, os
import customTorchTools
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Union




def ComputePoarNdirectionDistance(CconicMatrix:Union[np.array | torch.tensor | list], pointCoords: Union[np.array | torch.tensor], device=customTorchTools.getDevice()):
    '''
    Function to compute the Polar-n-direction distance of a point from a conic in the image plane represented by its [3x3] matrix.
    '''
    # Convert the input to homogeneous coordinates
    pointHomoCoords = torch.tensor([pointCoords[0], pointCoords[1], 1], shape=(3,1))

    # Compute auxiliary variables
    CbarMatrix = np.zeros((3,3))
    CbarMatrix[0:2,0:2] = CconicMatrix[0:2,0:2]

    Gmatrix = torch.matmul(CconicMatrix, CbarMatrix)
    Wmatrix = torch.matmul(torch.matmul(CbarMatrix.transpose, CconicMatrix), CbarMatrix)

    # Compute Gdist2, CWdist and Cdist
    Cdist = torch.matmul(pointHomoCoords.transpose() * torch.matmul(CconicMatrix, pointHomoCoords))
    
    Gdist = ( torch.matmul(pointHomoCoords.transpose() * torch.matmul(Gmatrix, pointHomoCoords)) )
    Gdist2 = Gdist * Gdist
    
    Wdist = ( torch.matmul(pointHomoCoords.transpose() * torch.matmul(Wmatrix, pointHomoCoords)) )

    CWdist = Cdist * Wdist

    # Compute the square distance depending on if condition
    if Gdist2 >= CWdist:

        sqrDist = Cdist / (Gdist * ( 1+torch.sqrt(1 + (Gdist2 - CWdist)/Gdist2) )**2)

    else:
        sqrDist = 0.25 * (Cdist**2 / Gdist)


    ##### DEBUG PRINTS ##########
    print('CbarMatrix:', CbarMatrix)
    print('Gmatrix:', Gmatrix)
    print('Wmatrix:', Wmatrix)
    ############################

    return sqrDist




def main():
    print("Test the Polar-n-direction distance function for the evaluation of the distance of a point from a conic in the image place represented by its [3x3] matrix.")

    # %% TEST: check which point matplotlib and openCV uses as origin of the image
    # TODO

    # %% TEST: loss function
    # Construct meshgrid over [7x7] patch for evaluation
    # TODO

    # Define the conic matrix

    # Evaluate the loss function over the meshgrid

    # Plot the loss function values




if __name__ == "__main__":
    main()