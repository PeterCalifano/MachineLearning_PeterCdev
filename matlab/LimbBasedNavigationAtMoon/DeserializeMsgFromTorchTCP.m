function [dPredictedPixCorrection] = DeserializeMsgFromTorchTCP(recvBytes, recvDataBuffer)%#codegen
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
% 17-06-2024        Pietro Califano         Adapted from script.
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code
% ui8flattenedWindow
% dRmoonDEM
% dSunDir_PixCoords
% dAttDCM_fromTFtoCAM
% dPosCam_TF
% ui8coarseLimbPixels

dataBufferReceived = typecast(recvDataBuffer, 'single');

fprintf('\nReceived data length: %d. \nData vector: ', recvBytes);
disp(dataBufferReceived);

dPredictedPixCorrection = double(dataBufferReceived);



end