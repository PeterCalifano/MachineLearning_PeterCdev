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

JSONdecoder();



%% Model loading
path2model;
trainedCNN = ImportModelFromONNx(path2model, 'regression');



%% Model evaluation     
saveID = 1;

    
% Get flattened patch
flattenedWindow = ui8flattenedWindows(:, sampleID);

% Validate patch counting how many pixels are completely black or white
% pathIsValid = customTorch.IsPatchValid(flattenedWindow, lowerIntensityThr=5);

% Compose input sample
inputDataSample = zeros(60, 1);

inputDataSample(1:49)   = flattenedWindow;
inputDataSample(50)     = dRmoonDEM;
inputDataSample(51:52)  = dSunDir_PixCoords;
inputDataSample(53:55)  = dAttDCM_fromTFtoCAM; % Convert Attitude matrix to MRP parameters
inputDataSample(56:58)  = dPosCam_TF;
inputDataSample(59:60)  = ui16coarseLimbPixels;

% TODO: check which shape the model from ONNx will require
% Which dtype? --> Python should use float (?)

outputPrediction = predict(trainedCNN, reshape(inputDataSample, 1, 60) );

