close all
clear
clc


addpath('..')

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
% -------------------------------------------------------------------------------------------------------------
% DEPENDENCIES
% Deep Learning toolbox
% -------------------------------------------------------------------------------------------------------------
% Future upgrades

%% OPTIONS



%% Simulation data loading
dataJSONpath = fullfile();
datastruct = JSONdecoder(dataJSONpath);

% Get flattened patch
sampleID = 1
flattenedWindow  = datastruct.ui8flattenedWindows(:, sampleID);
coarseLimbPixels = datastruct.ui16coarseLimbPixels(:, sampleID)
% Validate patch counting how many pixels are completely black or white
% pathIsValid = customTorch.IsPatchValid(flattenedWindow, lowerIntensityThr=5);

% Compose input sample
inputDataSample = zeros(60, 1, 'single');

inputDataSample(1:49)   = single(flattenedWindow);
inputDataSample(50)     = single(datastruct.dRmoonDEM);
inputDataSample(51:52)  = single(datastruct.dSunDir_PixCoords);
inputDataSample(53:55)  = single(quat2mrp( DCM2quat(datastruct.dAttDCM_fromTFtoCAM, false) )); % Convert Attitude matrix to MRP parameters
inputDataSample(56:58)  = single(datastruct.dPosCam_TF);
inputDataSample(59:60)  = single(coarseLimbPixels);

%% Model loading
path2model;
trainedCNN = ImportModelFromONNx(path2model, 'regression');

%% Model evaluation     

% NOTES: 
% 1) Check which shape the model from ONNx will require --> MATLAB converts the model as it is. 
%    Therefore it will requires the same shape as in Torch code.
% 2) Which dtype? --> Python should use float (?) --> Apparently yes, try single

outputPrediction = predict(trainedCNN, reshape(inputDataSample, 1, 60) );

