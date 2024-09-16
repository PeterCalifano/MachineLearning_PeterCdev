# Script created by PeterC 21-05-2024 to experiment with Limb-based navigation using CNN-NN network
# Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225
# Prototype architecture designed and coded, 03-05-2024

# BRIEF DESCRIPTION
# The network takes windows of Moon images where the illuminated limb is present, which may be a feature map already identifying the limb if edge detection has been performed.
# Convolutional layers process the image patch to extract features, then stacked with several other inputs downstream. Among these, the relative attitude of the camera wrt to the target body 
# and the Sun direction in the image plane. A series of fully connected layers then map the flatten 1D vector of features plus the contextual information to infer a correction of the edge pixels.
# This correction is intended to remove the effect of the imaged body morphology (i.e., the network will account for the DEM "backward") such that the refined edge pixels better adhere to the
# assumption of ellipsoidal/spherical body without terrain effects. The extracted pixels are then used by the Christian Robinson algorithm to perform derivation of the position vector.

# To insert multiple inputs at different layers of the network, following the discussion:
# 1) Construct the input tensor X such that it can be sliced into the image and the other data
# 2) Define the architecture class
# 3) In defining the forward(self, x) function, slice the x input according to the desired process, reshape and do other operations. 
# 4) Concatenate the removed data with the new inputs from the preceding layers as input to where they need to go.

# Import modules
import os
import matplotlib.pyplot as plt
import torch, sys, os
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
import pc_torchTools

import datetime
from torch import nn
from math import sqrt
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

from typing import Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim
import torch.nn.functional as torchFunc # Module to apply activation functions in forward pass instead of defining them in the model class

# Possible option: X given as input to the model is just one, but managed internally to the class, thus splitting the input as appropriate and only used in the desired layers.
# Alternatively, forward() method can accept multiple inputs based on the definition.

# DEVNOTE: check which modifications are needed for training in mini-batches
# According to GPT4o, the changes required fro the training in batches are not many. Simply build the dataset accordingly, so that Torch is aware of the split:
# Exameple: the forward() method takes two input vectors: forward(self, main_inputs, additional_inputs)
# main_inputs = torch.randn(1000, 784)  # Main input (e.g., image features)
# additional_inputs = torch.randn(1000, 10)  # Additional input (e.g., metadata)
# labels = torch.randint(0, 10, (1000,))
# dataset = TensorDataset(main_inputs, additional_inputs, labels)

# %% Auxiliary functions

def create_custom_mosaic_with_points_gray(image_array, positions, large_image_size, patch_points, global_points=None, save_path=None):
    # Get dimensions from the input array
    img_height, img_width, num_images = image_array.shape

    # Initialize the large image with a fixed size
    large_image = np.zeros(large_image_size, dtype=np.uint8)

    # Place each image in the large image at specified positions
    for idx in range(num_images):
        center_y, center_x = positions[idx]

        # Calculate the top-left corner from the center position
        start_y = int(center_y - np.floor(img_height / 2))
        start_x = int(center_x - np.floor(img_width / 2))

        # Ensure the patch fits within the large image dimensions
        if start_y >= 0 and start_x >= 0 and start_y + img_height <= large_image_size[0] and start_x + img_width <= large_image_size[1]:
            # Debug output
            print(f'Placing image {idx+1} at position ({start_y}, {start_x})')

            # Place the patch
            large_image[start_y:start_y+img_height,
                        start_x:start_x+img_width] = image_array[:, :, idx]
        else:
            print(
                f'Warning: Image {idx+1} at position ({start_y}, {start_x}) exceeds the large image bounds and will not be placed.')

    # Display the large image with patches
    plt.figure()
    plt.imshow(large_image, cmap='gray')
    plt.title('Custom Mosaic Image with Points')

    # Plot the points corresponding to each patch
    for idx in range(num_images):
        point_y, point_x = patch_points[idx]
        plt.plot(point_x, point_y, 'ro', markersize=5, linewidth=0.5)

    # Plot the additional global points if provided
    if global_points is not None:
        for idx in range(len(global_points)):
            point_y, point_x = global_points[idx]
            plt.plot(point_x, point_y, 'bo', markersize=10, linewidth=0.5)

    # Save the figure if save_path is provided
    if save_path is not None:
        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the image
        save_file = os.path.join(save_path, 'mosaic.png')
        plt.savefig(save_file, bbox_inches='tight')
        print(f'Mosaic saved to {save_file}')

    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: create a random array of images for demonstration
    height = 100
    width = 100
    num_images = 5
    image_array = np.random.randint(
        0, 256, (height, width, num_images), dtype=np.uint8)

    # Example positions (top-left corners) for each patch
    positions = np.array([[1, 1], [1, 150], [150, 1], [150, 150], [75, 75]])

    # Example points corresponding to each patch
    patch_points = np.array(
        [[50, 50], [50, 200], [200, 50], [200, 200], [125, 125]])

    # Example additional global points
    global_points = np.array([[30, 30], [70, 250], [250, 70], [270, 270]])

    # Size of the large image
    large_image_size = (300, 300)

    # Path to save the mosaic
    save_path = 'mosaic_images'

    create_custom_mosaic_with_points_gray(
        image_array, positions, large_image_size, patch_points, global_points, save_path)



# %% Custom loss function for Moon Limb pixel extraction CNN enhancer - 01-06-2024
def MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, paramsTrain:dict=None, paramsEval:dict=None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    if paramsTrain is None:
        coeff = 0.5
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']
        
    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    # Evaluate loss
    conicLoss = 0.0 # Weighting violation of Horizon conic equation
    L2regLoss = 0.0

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=pc_torchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

        #L2regLoss += torch.norm(predictCorrection[idBatch]) # Weighting the norm of the correction to keep it as small as possible

    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss
    return lossValue

# %% Custom loss function for Moon Limb pixel extraction CNN enhancer with strong loss for out-of-patch predictions - 23-06-2024
def MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm(predictCorrection, labelVector, paramsTrain:dict=None, paramsEval:dict=None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss

    if paramsTrain is None:
        coeff = 0.5
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']

    # Temporary --> should come from paramsTrain dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    if 'RectExpWeightCoeff' in paramsTrain.keys():
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']
    else:
        RectExpWeightCoeff = 1

    # Evaluate loss
    outOfPatchoutLoss = 0.0
    conicLoss = 0.0 # Weighting violation of Horizon conic equation
    L2regLoss = 0.0

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel and conic loss
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=pc_torchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix))

        # Add average of the two coordinates to the total cost term
        outOfPatchoutLoss += outOfPatchoutLoss_Quadratic(predictCorrection[idBatch, :].reshape(2,1), halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)

        #L2regLoss += torch.norm(predictCorrection[idBatch]) # Weighting the norm of the correction to keep it as small as possible

    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss
    
    return lossValue

# %% Additional Loss term for out-of-patch predictions based on Rectified Exponential function - 23-06-2024
def outOfPatchoutLoss_RectExp(predictCorrection, halfPatchSize=3.5, slopeMultiplier=2):
    if predictCorrection.size()[0] != 2:
        raise ValueError('predictCorrection must have 2 rows for x and y pixel correction')
    
    # Compute the out-of-patch loss 
    numOfCoordsOutOfPatch = 1
    tmpOutOfPatchoutLoss = 0.0

    # Compute the out-of-patch loss
    if abs(predictCorrection[0]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.exp(slopeMultiplier*(predictCorrection[0]**2 - halfPatchSize**2))

    if abs(predictCorrection[1]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.exp(slopeMultiplier*(predictCorrection[1]**2 - halfPatchSize**2))
        numOfCoordsOutOfPatch += 1

    if tmpOutOfPatchoutLoss > 0:
        if tmpOutOfPatchoutLoss.isinf():
            tmpOutOfPatchoutLoss = 1E4

    return tmpOutOfPatchoutLoss/numOfCoordsOutOfPatch # Return the average of the two losses

# %% Additional Loss term for out-of-patch predictions based on quadratic function - 25-06-2024
def outOfPatchoutLoss_Quadratic(predictCorrection, halfPatchSize=3.5, slopeMultiplier=0.2):
    if predictCorrection.size()[0] != 2:
        raise ValueError('predictCorrection must have 2 rows for x and y pixel correction')
    
    # Compute the out-of-patch loss 
    numOfCoordsOutOfPatch = 1
    tmpOutOfPatchoutLoss = 0.0

    # Compute the out-of-patch loss
    if abs(predictCorrection[0]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.square(slopeMultiplier*(predictCorrection[0] - halfPatchSize)**2)

    if abs(predictCorrection[1]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.square(slopeMultiplier*(predictCorrection[1] - halfPatchSize)**2)
        numOfCoordsOutOfPatch += 1

    if tmpOutOfPatchoutLoss > 0:
        if tmpOutOfPatchoutLoss.isinf():
            raise ValueError('tmpOutOfPatchoutLoss is infinite')
        
    return tmpOutOfPatchoutLoss/numOfCoordsOutOfPatch # Return the average of the two losses


def outOfPatchoutLoss_Quadratic_asTensor(predictCorrection:torch.tensor, halfPatchSize=3.5, slopeMultiplier=0.2):

    if predictCorrection.size()[1] != 2:
        raise ValueError('predictCorrection must have 2 rows for x and y pixel correction')
    
    device = predictCorrection.device
    batchSize = predictCorrection.size()[0]

    numOfCoordsOutOfPatch = torch.ones(batchSize, 1, device=device)

    # Check which samples have the coordinates out of the patch
    idXmask = abs(predictCorrection[:, 0]) > halfPatchSize
    idYmask = abs(predictCorrection[:, 1]) > halfPatchSize

    numOfCoordsOutOfPatch += idYmask.view(batchSize, 1)

    tmpOutOfPatchoutLoss = torch.zeros(batchSize, 1, device=device)

    # Add contribution for all X coordinates violating the condition
    tmpOutOfPatchoutLoss[idXmask] += torch.square(slopeMultiplier*(predictCorrection[idXmask, 0] - halfPatchSize)**2).reshape(-1, 1)
    # Add contribution for all Y coordinates violating the condition
    tmpOutOfPatchoutLoss[idYmask] += torch.square(slopeMultiplier*(predictCorrection[idYmask, 1] - halfPatchSize)**2).reshape(-1, 1)

    if any(tmpOutOfPatchoutLoss > 0):
        if any(tmpOutOfPatchoutLoss.isinf()):
            raise ValueError('tmpOutOfPatchoutLoss is infinite')
        
    return torch.div(tmpOutOfPatchoutLoss, numOfCoordsOutOfPatch) # Return the average of the two losses for each entry in the batch

#######################################################################################################
# %% Custom normalized loss function for Moon Limb pixel extraction CNN enhancer - 23-06-2024
def MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm(predictCorrection, labelVector, paramsTrain:dict=None, paramsEval:dict=None):

    # Get parameters and labels for computation of the loss
    if paramsTrain is None:
        coeff = 1
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']

    if paramsTrain is None:
        RectExpWeightCoeff = 1
    elif 'RectExpWeightCoeff' in paramsTrain.keys(): 
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']


    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:11]
    baseCost2 = labelVector[:, 11]

    # Evaluate loss
    normalizedConicLoss = 0.0 # Weighting violation of Horizon conic equation
    outOfPatchoutLoss = 0.0

    batchSize = labelVector.size()[0]

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor([0,0,1], dtype=torch.float32, device=pc_torchTools.GetDevice()).reshape(3,1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(2,1) + predictCorrection[idBatch, :].reshape(2,1)

        normalizedConicLoss += ((torch.matmul(correctedPix.T, torch.matmul(LimbConicMatrixImg[idBatch, :, :].reshape(3,3), correctedPix)))**2/baseCost2[idBatch])

        # Add average of the two coordinates to the total cost term
        outOfPatchoutLoss += outOfPatchoutLoss_Quadratic(predictCorrection[idBatch, :].reshape(2,1), halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)
    
    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * (normalizedConicLoss/batchSize) + (1-coeff) * L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss

    return lossValue

#######################################################################################################
# %% Function to validate path (check it is not completely black or white)
def IsPatchValid(patchFlatten, lowerIntensityThr=3):
    
    # Count how many pixels are below threshold
    howManyBelowThreshold = np.sum(patchFlatten <= lowerIntensityThr)
    howManyPixels = len(patchFlatten)
    width = np.sqrt(howManyPixels)
    lowerThreshold = width/2
    upperThreshold = howManyPixels - lowerThreshold
    if howManyBelowThreshold <  lowerThreshold or howManyBelowThreshold > upperThreshold:
        return False
    else:
        return True


# %% Custom normalized loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm_asTensor(predictCorrection, labelVector, params:dict=None):

    # Get parameters and labels for computation of the loss
    TrainingMode = params.get('TrainingMode', True)
    paramsTrain = params.get('paramsTrain', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 1})
    paramsEval = params.get('paramsEval', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 0})

    if TrainingMode:
        RectExpWeightCoeff = paramsTrain.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsTrain.get('ConicLossWeightCoeff', 1)
    else:
        RectExpWeightCoeff = paramsEval.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsEval.get('ConicLossWeightCoeff', 1)

    
    # Temporary --> should come from params dictionary
    patchSize = params.get('patchSize', 7)
    slopeMultiplier = params.get('slopeMultiplier', 2)
    halfPatchSize = patchSize/2


    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = labelVector.device
    
    LimbConicMatrixImg = torch.tensor((labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T, dtype=torch.float32, device=device)

    patchCentre = labelVector[:, 9:11]
    baseCost2 = labelVector[:, 11]

    # Evaluate loss terms
    #normalizedConicLoss = torch.zeros(batchSize, 1, 1, dtype=torch.float32, device=device) # Weighting violation of Horizon conic equation
    #outOfPatchoutLoss = torch.zeros(batchSize, 1, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(batchSize, 3, 1, dtype=torch.float32, device=device)

    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    normalizedConicLoss = torch.div((torch.bmm(correctedPix.transpose(1,2), torch.bmm(LimbConicMatrixImg, correctedPix)))**2, baseCost2.reshape(batchSize, 1,1))

    # Add average of the two coordinates to the total cost term
    outOfPatchoutLoss = outOfPatchoutLoss_Quadratic_asTensor(predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier).reshape(batchSize, 1, 1)

    if ConicLoss2WeightCoeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = ConicLoss2WeightCoeff * (normalizedConicLoss.sum()) + (1-ConicLoss2WeightCoeff) * L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss.sum()

    # Return sum of loss for the whole batch
    return lossValue/batchSize


# %% Polar-n-direction loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_PolarNdirectionDistanceWithOutOfPatch_asTensor(predictCorrection, labelVector, paramsTrain:dict=None, paramsEval:dict=None):
        # Get parameters and labels for computation of the loss

    if paramsTrain is None:
        RectExpWeightCoeff = 1
    elif 'RectExpWeightCoeff' in paramsTrain.keys(): 
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']

    slopeMultiplier = 4

    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = labelVector.device
    
    LimbConicMatrixImg = torch.tensor((labelVector[:, 0:9].T).reshape(3, 3, labelVector.size()[0]).T, dtype=torch.float32, device=device)
    patchCentre = labelVector[:, 9:11]

    # Evaluate loss terms
    outOfPatchoutLoss = torch.zeros(batchSize, 1, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(batchSize, 3, 1, dtype=torch.float32, device=device)

    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    # Compute the Polar-n-direction distance
    polarNdirectionDist = ComputePolarNdirectionDistance_asTensor(LimbConicMatrixImg, correctedPix)

    # Add average of the two coordinates to the total cost term
    outOfPatchoutLoss += outOfPatchoutLoss_Quadratic_asTensor(predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier).reshape(batchSize, 1, 1)

    # Total loss function
    lossValue = polarNdirectionDist + RectExpWeightCoeff * outOfPatchoutLoss

    # Return sum of loss for the whole batch
    return torch.sum(lossValue/batchSize)
    
    

def ComputePolarNdirectionDistance_asTensor(CconicMatrix:Union[np.array , torch.tensor , list], 
                                   pointCoords: Union[np.array , torch.tensor]):
    '''
    Function to compute the Polar-n-direction distance of a point from a conic in the image plane represented by its [3x3] matrix using torch tensors operations.
    '''
    device = pointCoords.device
    # Shape point coordinates as tensor
    batchSize = pointCoords.shape[0]

    pointHomoCoords_tensor = torch.zeros((batchSize,3,1), dtype=torch.float32, device=device)
    pointHomoCoords_tensor[:, 0:2, 0] = torch.stack((pointCoords[:, 0, 0], pointCoords[:, 1, 0]), dim=1).reshape(batchSize, 2)
    pointHomoCoords_tensor[:, 2, 0] = 1

    # Reshape Conic matrix to tensor
    CconicMatrix = CconicMatrix.view(batchSize, 3, 3).to(device)

    CbarMatrix_tensor = torch.zeros((batchSize, 3, 3), dtype=torch.float32, device=device)
    CbarMatrix_tensor[:, 0:2,0:3] = CconicMatrix[:, 0:2,0:3]

    Gmatrix_tensor = torch.bmm(CconicMatrix, CbarMatrix_tensor)
    Wmatrix_tensor = torch.bmm(torch.bmm(CbarMatrix_tensor.transpose(1,2), CconicMatrix), CbarMatrix_tensor)

    # Compute Gdist2, CWdist and Cdist
    Cdist_tensor = torch.bmm( pointHomoCoords_tensor.transpose(1, 2), torch.bmm(CconicMatrix, pointHomoCoords_tensor) ) 
    
    Gdist_tensor = torch.bmm( pointHomoCoords_tensor.transpose(1, 2), torch.bmm(Gmatrix_tensor, pointHomoCoords_tensor) )
    Gdist2_tensor = torch.bmm(Gdist_tensor, Gdist_tensor)
    
    Wdist_tensor = torch.bmm(pointHomoCoords_tensor.transpose(1, 2), torch.bmm(Wmatrix_tensor, pointHomoCoords_tensor)) 
    CWdist_tensor = torch.bmm(Cdist_tensor, Wdist_tensor)


    # Get mask for the condition
    idsMask = Gdist2_tensor >= CWdist_tensor

    notIdsMask = Gdist2_tensor < CWdist_tensor

    # Compute the square distance depending on if condition
    sqrDist_tensor = torch.zeros(batchSize, dtype=torch.float32, device=device)

    sqrDist_tensor[idsMask[:,0,0]] = Cdist_tensor[idsMask] / ( Gdist_tensor[idsMask] * ( 1 + torch.sqrt(1 + (Gdist2_tensor[idsMask] - CWdist_tensor[idsMask]) / Gdist2_tensor[idsMask]) )**2)
    sqrDist_tensor[notIdsMask[:,0,0]] = 0.25 * (Cdist_tensor[notIdsMask]**2 / Gdist_tensor[notIdsMask])

    # Return mean over the whole batch
    return abs(sqrDist_tensor)


# %% MSE + Conic Loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor(predictCorrection, labelVector, 
                                                                            params:dict=None):
    
    # Get parameters and labels for computation of the loss
    TrainingMode = params.get('TrainingMode', True)
    paramsTrain = params.get('paramsTrain', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 1})
    paramsEval = params.get('paramsEval', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 0})

    if TrainingMode:
        RectExpWeightCoeff = paramsTrain.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsTrain.get('ConicLossWeightCoeff', 1)
    else:
        RectExpWeightCoeff = paramsEval.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsEval.get('ConicLossWeightCoeff', 1)

    
    # Temporary --> should come from params dictionary
    patchSize = params.get('patchSize', 7)
    slopeMultiplier = params.get('slopeMultiplier', 2)
    halfPatchSize = patchSize/2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = predictCorrection.device
    
    # Step 1: Select the first 9 columns for all rows
    # Step 2: Permute the dimensions to match the transposition (swap axes 0 and 1)
    # Step 3: Reshape the permuted tensor to the specified dimensions
    # Step 4: Permute again to match the final transposition (swap axes 0 and 2)

    LimbConicMatrixImg = ((labelVector[:, 0:9].permute(1, 0)).reshape(3, 3, labelVector.size()[0]).permute(2, 0, 1)).clone().to(device)

    assert(LimbConicMatrixImg.shape[0] == batchSize)


    patchCentre      = labelVector[:, 9:11]  # Patch centre coordinates (pixels)
    baseCost2        = labelVector[:, 11]    # Base cost for the conic constraint evaluated at patch centre
    targetPrediction = labelVector[:, 12:14] # Target prediction for the pixel correction

    # Initialize arrays for the loss terms
    #normalizedConicLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device) # Weighting violation of Horizon conic equation
    #outOfPatchoutLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(batchSize, 3, 1, dtype=torch.float32, device=device)
    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    # Compute the normalized conic loss
    if ConicLoss2WeightCoeff != 0:
        #normalizedConicLoss = torch.div( ((torch.bmm(correctedPix.transpose(1,2), torch.bmm(LimbConicMatrixImg, correctedPix)) )**2).reshape(batchSize, 1), baseCost2.reshape(batchSize, 1))
        unnormalizedConicLoss = ( (torch.bmm(correctedPix.transpose(1, 2), torch.bmm( LimbConicMatrixImg, correctedPix)))**2 ).reshape(batchSize, 1)

    else:
        unnormalizedConicLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device)

    # Compute the MSE loss term
    mseLoss = torch.nn.functional.mse_loss(correctedPix[:, 0:2, 0], targetPrediction, size_average=None, reduce=None, reduction='mean')

    if RectExpWeightCoeff != 0:
        # Add average of the two coordinates to the out of patch cost term
        outOfPatchLoss = outOfPatchoutLoss_Quadratic_asTensor(predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)
    else:
        outOfPatchLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device)

    # Total loss function
    normalizedConicLossTerm = ConicLoss2WeightCoeff * torch.sum(unnormalizedConicLoss)/batchSize
    outOfPatchLossTerm = torch.sum((RectExpWeightCoeff * outOfPatchLoss))/batchSize
    lossValue =  normalizedConicLossTerm + mseLoss + outOfPatchLossTerm

    # Return sum of loss for the whole batch
    return {'lossValue': lossValue, 'normalizedConicLoss': normalizedConicLossTerm, 'mseLoss': mseLoss, 'outOfPatchoutLoss': outOfPatchLossTerm}


# %% ARCHITECTURES ############################################################################################################

def AutoComputeConvBlocksOutput(self, kernelSizes:list, poolingKernelSize:list=None):
        # NOTE: stride and padding are HARDCODED in this version
        # Automatically compute the number of features from the last convolutional layer (flatten of the volume)
        outputMapSize = [self.patchSize, self.patchSize]

        if poolingKernelSize is None:
            poolingKernelSize = list(np.ones(len(kernelSizes)))
        
        assert(self.numOfConvLayers == len(kernelSizes) == len(poolingKernelSize))

        for idL in range(self.numOfConvLayers):

            convBlockOutputSize = pc_torchTools.ComputeConvBlockOutputSize(outputMapSize, self.outChannelsSizes[idL], kernelSizes[idL], poolingKernelSize[idL], 
                                                                              convStrideSize=1, poolingStrideSize=poolingKernelSize[idL],
                                                                            convPaddingSize=0, poolingPaddingSize=0)
            
            print(('Output size of ConvBlock ID: {ID}: {outSize}').format(ID=idL, outSize=convBlockOutputSize))
            # Get size from previous convolutional block
            outputMapSize[0] = convBlockOutputSize[0][0]
            outputMapSize[1] = convBlockOutputSize[0][1]

        return convBlockOutputSize

# %% Custom CNN-NN model for Moon limb pixel extraction enhancer - 01-06-2024
'''Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
'''
class HorizonExtractionEnhancerCNNv1avg(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]

        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]


        # Convolutional layers
        # L1 (Input)
        val = self.avgPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.avgPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2 (with target average radius in pixels) - 21-06-2024
class HorizonExtractionEnhancerCNNv2avg(nn.Module):
    '''
    Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
    '''
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.avgPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.avgPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
# %% Custom CNN-NN model for Moon limb pixel extraction enhancer V1max - 23-06-2024
    '''Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
    '''
class HorizonExtractionEnhancerCNNv1max(nn.Module):

    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]

        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # DEBUG
        #print(img2Dinput[0, 0, :,:])
        ########################################

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2max (with target average radius in pixels) - 23-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv2max(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
