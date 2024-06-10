close all
clear
clc

addpath('..');
addpath('testModels\');
addpath(genpath('testDatapairs\'));
% SCRIPT NAME
% LimbPixelsEnhancerCNN_Testing
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the script does
% -------------------------------------------------------------------------------------------------------------
% NEEDED FROM BASE WORKSPACE
% in1 [dim] description
% -------------------------------------------------------------------------------------------------------------
% OUT TO BASE WORKSPACE
% out1 [dim] description
% -------------------------------------------------------------------------------------------------------------
% CHANGELOG
% 09-06-2024    Pietro Califano     Script initialized. Model loading code added.
% 10-06-2024    Pietro Califano     First version of script completed.
% -------------------------------------------------------------------------------------------------------------
% DEPENDENCIES
% Deep Learning toolbox
% -------------------------------------------------------------------------------------------------------------
% Future upgrades

%% OPTIONS
fileID   = 1;
sampleID = 1; % Sample patch ID in loaded datapair file
testDatapairsStruct = getDirStruct(dataJSONpath);

dataJSONpath = fullfile(testDatapairsStruct(fileID));

%% Simulation data loading
datastruct = JSONdecoder(dataJSONpath(fileID));

% Get flattened patch
flattenedWindow  = datastruct.ui8flattenedWindows(:, sampleID);
coarseLimbPixels = datastruct.ui16coarseLimbPixels(:, sampleID)
% Validate patch counting how many pixels are completely black or white
% pathIsValid = customTorch.IsPatchValid(flattenedWindow, lowerIntensityThr=5);

% Improvements I should do to the model:
% 1) Remove constant input R_DEM. If needed, convert it to apparent pixel size to introduce the concept of
%   "distance" in the network map. However, note that this is already present due to the relative position
%   vector. Remove it unless generalization to other targets is required.
% 2) Input image should be normalized wrt to the maximum. This tip applies to all the inputs.

% Compose input sample
inputDataSample = zeros(60, 1, 'single');

inputDataSample(1:49)  = single(flattenedWindow); 
inputDataSample(50)    = single(datastruct.dRmoonDEM);
inputDataSample(51:52) = single(datastruct.dSunDir_PixCoords);
inputDataSample(53:55) = single(quat2mrp( DCM2quat(reshape(datastruct.dAttDCM_fromTFtoCAM, 3, 3), false) )); % Convert Attitude matrix to MRP parameters
inputDataSample(56:58) = single(datastruct.dPosCam_TF);
inputDataSample(59:60) = single(coarseLimbPixels);

%% Model loading
path2model = 'testModels';
modelName = 'traineModel_0';
trainedCNN = ImportModelFromONNx(fullfile(path2model, modelName), 'regression');

% Print model summary 
summary(trainedCNN);

% Plot model architecture
figure(1);
plot(trainedCNN);

%% Model evaluation     

% NOTES: 
% 1) Check which shape the model from ONNx will require --> MATLAB converts the model as it is. 
%    Therefore it will require the same shape as in Torch code.
% 2) Which dtype? --> Python should use float (?) --> Apparently yes, try single()

outputPrediction = predict(trainedCNN, reshape(inputDataSample, 1, 60) );

