function [dPredictedPixCorrection] = callModelTorchTCP(ui8flattenedWindow, dRmoonDEM, ...
    dSunDir_PixCoords, dAttMRP_fromTFtoCAM, dPosCam_TF, ui8coarseLimbPixels, commHandler)
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
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code
 
% ui8flattenedWindow  = inputDataStruct.ui8flattenedWindow;
% dRmoonDEM           = inputDataStruct.dRmoonDEM;
% dSunDir_PixCoords   = inputDataStruct.dSunDir_PixCoords;
% dAttMRP_fromTFtoCAM = inputDataStruct.dAttMRP_fromTFtoCAM;
% dPosCam_TF          = inputDataStruct.dPosCam_TF;
% ui8coarseLimbPixels = inputDataStruct.ui8coarseLimbPixels;

% Serialize message to send with input sample for CNN
[~, dataBufferToWrite] = SerializeMsgToTorchTCP(ui8flattenedWindow, ...
    dRmoonDEM, ...
    dSunDir_PixCoords, ...
    dAttMRP_fromTFtoCAM, ...
    dPosCam_TF, ...
    ui8coarseLimbPixels);

% Send buffer to server
commHandler.WriteBuffer(dataBufferToWrite);

% Read buffer from server
[recvBytes, recvDataBuffer] = commHandler.ReadBuffer();

% Deserialize data from server to get CNN prediction
[dPredictedPixCorrection] = DeserializeMsgFromTorchTCP(recvBytes, recvDataBuffer);


end
