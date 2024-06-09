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



%% Model loading

path2model;
trainedCNN = ImportModelFromONNx(path2model, 'regression');







