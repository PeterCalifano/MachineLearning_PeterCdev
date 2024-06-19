close all
clear
clc

addpath('..');
addpath('testModelsONNx');
addpath(genpath('testDatapairs'));
addpath(genpath('/home/peterc/devDir/MATLABcodes'))
addpath('/home/peterc/devDir/robots-api/matlab/CommManager')

% SCRIPT NAME
% LimbPixelsEnhancerCNN_Testing
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the script does
% -------------------------------------------------------------------------------------------------------------
% CHANGELOG
% 09-06-2024    Pietro Califano     Script initialized. Model loading code added.
% 10-06-2024    Pietro Califano     First version of script completed.
% 15-06-2024    Pietro Califano     Added code to use Torch over TCP server. Test passed.
% 19-06-2024    Pietro Califano     Updated Torch Model to v2.0
% -------------------------------------------------------------------------------------------------------------
% DEPENDENCIES
% Deep Learning toolbox
% -------------------------------------------------------------------------------------------------------------

%% OPTIONS
bUSE_TORCH_OVER_TCP = true;
bUSE_PYENV = true;
bRUN_SIMULINK_SIM = true;

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
inputDataSample = zeros(56, 1, 'single');


% Assign labels to labels data array
ptrToInput = 1;

flattenedWindSize = length(flattenedWindow);

inputDataSample(ptrToInput:ptrToInput+flattenedWindSize-1)  = single(flattenedWindow);

% Update index
ptrToInput = ptrToInput + flattenedWindSize; 

inputDataSample(ptrToInput:ptrToInput+length(datastruct.metadata.dSunDir_PixCoords)-1) = single(datastruct.metadata.dSunDir_PixCoords);

% Update index
ptrToInput = ptrToInput + length(datastruct.metadata.dSunDir_PixCoords); % Update index

tmpVal = quat2mrp( DCM2quat(reshape(datastruct.metadata.dAttDCM_fromTFtoCAM, 3, 3), false) );

inputDataSample(ptrToInput : ptrToInput + length(tmpVal)-1) = single(tmpVal); % Convert Attitude matrix to MRP parameters

% Update index
ptrToInput = ptrToInput + length(tmpVal); % Update index

inputDataSample(ptrToInput:end) = single(coarseLimbPixels);


if bUSE_TORCH_OVER_TCP == true

    % Define object properties
    serverAddress = '127.0.0.1';
    portNumber = uint16(50000);
    SEND_SHUTDOWN = false;

    dataBufferSize = uint16(60*8);
    
    % Create communication handler
    commHandler = CommManager(serverAddress, portNumber, 20);
    % Initialize tcpclient object and communication to server
    commHandler.Initialize()
    
    % Serialize input data from array
    if SEND_SHUTDOWN == true
        dataMessage = 'shutdown';
        if isa(dataMessage, 'string')
            dataMessage = char(dataMessage);
        end
        dataLength = length(dataMessage);
        lengthAsBytes = typecast(uint32(dataLength), 'uint8');
        dataBufferToWrite = [lengthAsBytes, uint8(dataMessage)];
    else
        if iscolumn(inputDataSample)
            dataMessage = transpose(inputDataSample);
        else
            dataMessage = (inputDataSample);
        end
        dataBuffer = typecast(dataMessage, 'uint8');
        dataLength = typecast(uint32(length(dataBuffer)), 'uint8');
        dataBufferToWrite = [dataLength, dataBuffer];
        % dataBufferToWrite_Size = length(dataBufferToWrite);
    end

    % Send data to server
    commHandler.WriteBuffer(dataBufferToWrite);
    if SEND_SHUTDOWN == true
        return;
    end
    % Test function to read data buffer
    % [recvBytes, recvDataBuffer, commManager] = ReadBuffer(commManager)
    [recvBytes, recvDataBuffer] = commHandler.ReadBuffer();
    % Convert received bytes stream into matrix
    dataBufferReceived = typecast(recvDataBuffer, 'single');
    fprintf('\nReceived data length: %d. \nData vector: ', recvBytes);
    disp(dataBufferReceived);

else
    % Python API environment setup
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

        elseif strcmpi(whoamiOut, 'peterc') && strcmpi(hostnameOut, 'peterc-MS-7916')
            PYTHONHOME = '/home/peterc/devDir/MachineLearning_PeterCdev/.venvML/bin/python3.10';

        elseif strcmpi(whoamiOut, 'peterc-flip\pietr')
            PYTHONHOME = "C:\devDir\MachineLearning_PeterCdev\.venvML\bin\python";
        end

        pyenvir = pyenv(Version=PYTHONHOME, ExecutionMode="OutOfProcess");
        disp(pyenvir);
    end

    % Model loading
    path2model = '/home/peterc/devDir/MachineLearning_PeterCdev';
    modelName = 'trainedTracedModel025_cpu.pt';
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

    % Prepare numpy array input
    inputSample_numpyInstance = np.ndarray(inputDataSample);

    % Perform inference
    modelPrediction = torchModelWrapper.forward(inputSample_numpyInstance);

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

end


%% TEST using Simulink model
modelName = 'testTorchTCP';
load_system(modelName);
open(modelName);

finalTime = 10;
timeStep = 1;


if bRUN_SIMULINK_SIM == true                        

    % Prepare input bus
    inputDataStruct.ui8flattenedWindow  = flattenedWindow;
    % inputDataStruct.dRmoonDEM           = datastruct.metadata.dRmoonDEM;
    inputDataStruct.dSunDir_PixCoords   = datastruct.metadata.dSunDir_PixCoords;
    inputDataStruct.dAttDCM_fromTFtoCAM = quat2mrp( DCM2quat(reshape(datastruct.metadata.dAttDCM_fromTFtoCAM, 3, 3), false) );
    % inputDataStruct.dPosCam_TF          = datastruct.metadata.dPosCam_TF;
    inputDataStruct.ui8coarseLimbPixelsdataBufferToWrite = coarseLimbPixels;

    timeGrid = 1:timeStep:finalTime;
    inputBusStruct = Simulink.Bus.createObject(inputDataStruct); %#ok<NASGU>
    InputTorchModelBus = slBus1;
    clear('inputBusStruct');

    % Create simulation object
    simInputObj = Simulink.SimulationInput(modelName);
    % Set model parameters
    simInputObj = simInputObj.setModelParameter('StopTime', num2str(finalTime));
    simInputObj = simInputObj.setModelParameter('FixedStep', num2str(timeStep));
    % Open model and simulate single run
    simOutputObj = sim(simInputObj);
    simOutputObj.SimulationMetadata.TimingInfo

end

