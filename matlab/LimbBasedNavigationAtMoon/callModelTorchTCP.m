function [dPredictedOutput] = callModelTorchTCP(i_strInputSamplesBatchStruct, ...
    i_ui32Nbatches, ...
    i_ui32InputSize, ... 
    i_ui32outputSize, ...
    commHandler)
%% PROTOTYPE
% [dPredictedPixCorrection] = callModelTorchTCP(i_strInputSamplesBatchStruct, ...
%     i_ui32Nbatches, ...
%     i_ui32InputSize, ... 
%     i_ui32outputSize, ...
%     commHandler)
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the function does
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% in1 [dim] description
% Name1                     []
% Name2                     []
% Name3                     []
% Name4                     []
% Name5                     []
% Name6                     []
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% out1 [dim] description
% Name1                     []
% Name2                     []
% Name3                     []
% Name4                     []
% Name5                     []
% Name6                     []
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 17-06-2024        Pietro Califano         Function 1st version.
% 20-06-2024        Pietro Califano         Updated version with reduced input size
% 25-06-2024        Pietro Califano         Modified to make function agnostic to input data
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code
 
% ui8flattenedWindow  = inputDataStruct.ui8flattenedWindow;
% dSunDir_PixCoords   = inputDataStruct.dSunDir_PixCoords;
% dAttMRP_fromTFtoCAM = inputDataStruct.dAttMRP_fromTFtoCAM;
% ui8coarseLimbPixels = inputDataStruct.ui8coarseLimbPixels;

% Serialize message to send with input sample for CNN

[~, dataBufferToWrite] = SerializeBatchMsgToTorchTCP(i_strInputSamplesBatchStruct, ...
    i_ui32Nbatches, ...
    i_ui32InputSize);

% Send buffer to server
commHandler.WriteBuffer(dataBufferToWrite, false);

% Read buffer from server
[recvBytes, recvDataBuffer] = commHandler.ReadBuffer();

% Deserialize data from server to get CNN prediction
[dPredictedOutput] = DeserializeMsgFromTorchTCP(recvBytes, recvDataBuffer, i_ui32outputSize);


end
