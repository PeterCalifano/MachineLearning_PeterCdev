close all
clear
clc

addpath('..');
addpath('testModelsONNx');
addpath(genpath('testDatapairs'));
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
bUSE_PYENV = true;

[~, whoamiOut] = system('whoami');
if not(strcmpi(computer, 'PCWIN64'))
    [outflag, hostnameOut] = system('cat /etc/hostname');
    hostnameOut = hostnameOut(1:end-1);

end

whoamiOut = whoamiOut(1:end-1);

if bUSE_PYENV 

    if strcmpi(whoamiOut, 'peterc') && strcmpi(hostnameOut, 'PETERC-FLIP')
        % if strcmpi(outToShell, 'PCWIN64')
        PYTHONHOME = '/home/peterc/devDir/MachineLearning_PeterCdev/.venvML/bin/python3.10';
    elseif strcmpi(whoamiOut, 'peterc-flip\pietr')
        PYTHONHOME = "C:\devDir\MachineLearning_PeterCdev\.venvML\bin\python";
    end

    pyenvir = pyenv(Version=PYTHONHOME, ExecutionMode="OutOfProcess");
    disp(pyenvir);
end

datasetID = 1;
fileID   = 1;
sampleID = 1; % Sample patch ID in loaded datapair file
testDatapairsStruct = getDirStruct('testDatapairs');
testDatapairsStruct = getDirStruct( fullfile('testDatapairs', testDatapairsStruct(datasetID).name) );

% Get path to JSON file containing data
dataJSONPath = fullfile(testDatapairsStruct(fileID).folder, testDatapairsStruct(fileID).name);

%% Simulation data loading
datastruct = JSONdecoder(dataJSONPath);

% Get flattened patch
flattenedWindow  = datastruct.ui8flattenedWindows(:, sampleID);
coarseLimbPixels = datastruct.ui16coarseLimbPixels(:, sampleID);
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
inputDataSample(50)    = single(datastruct.metadata.dRmoonDEM);
inputDataSample(51:52) = single(datastruct.metadata.dSunDir_PixCoords);
inputDataSample(53:55) = single(quat2mrp( DCM2quat(reshape(datastruct.metadata.dAttDCM_fromTFtoCAM, 3, 3), false) )); % Convert Attitude matrix to MRP parameters
inputDataSample(56:58) = single(datastruct.metadata.dPosCam_TF);
inputDataSample(59:60) = single(coarseLimbPixels);

%% Model loading
path2model = '/home/peterc/devDir/MachineLearning_PeterCdev';
modelName = 'trainedTracedModel075.pt';
path2customTorch = '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/';


if count(py.sys.path, path2customTorch) == 0
    insert(py.sys.path,int32(0), path2customTorch);
end

np = py.importlib.import_module('numpy');

customTorch = py.importlib.import_module('customTorch');

% Initialize MATLAB Torch wrapper object
% class TorchModel_MATLABwrap():
%     def __init__(self, trainedModelName:str, trainedModelPath:str)
%     def forward(self, inputSample:np.ndarray)

torchModelWrapper = customTorch.TorchModel_MATLABwrap(modelName, path2model);
return
% Prepare numpy array input
inputSample_numpyInstance = np.ndarray(inputDataSample);

% Perform inference
modelPrediction = torchModelWrapper.forward(inputSample_numpyInstance);

return 

%% Using import from ONNx to MATLAB Deep Learning toolbox
% NOTES: 
% 1) Check which shape the model from ONNx will require --> MATLAB converts the model as it is. 
%    Therefore it will require the same shape as in Torch code.
% 2) Which dtype? --> Python should use float (?) --> Apparently yes, try single()

% trainedCNN = ImportModelFromONNx(fullfile(path2model, modelName), 'regression');
% Print model summary 
% summary(trainedCNN);

% Plot model architecture
% figure(1);
% plot(trainedCNN);

% outputPrediction = predict(trainedCNN, reshape(inputDataSample, 1, 60) );

