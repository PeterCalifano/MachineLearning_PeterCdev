function [dPredictedPixCorrection] = callModelTorchTCP(ui8flattenedWindow, ...
    dSunDir_PixCoords, dAttMRP_fromTFtoCAM, ui8coarseLimbPixels, commHandler)
%% PROTOTYPE
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
% 20-06-2024        PIetro Califano         Updated version with reduced input size
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
[~, dataBufferToWrite] = SerializeMsgToTorchTCP(ui8flattenedWindow, ...
    dSunDir_PixCoords, ...
    dAttMRP_fromTFtoCAM, ...
    ui8coarseLimbPixels);

% Send buffer to server
commHandler.WriteBuffer(dataBufferToWrite);

% Read buffer from server
[recvBytes, recvDataBuffer] = commHandler.ReadBuffer();

% Deserialize data from server to get CNN prediction
[dPredictedPixCorrection] = DeserializeMsgFromTorchTCP(recvBytes, recvDataBuffer);


end
