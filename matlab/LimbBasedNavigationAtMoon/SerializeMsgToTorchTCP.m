function [dataLength, dataBufferToWrite] = SerializeMsgToTorchTCP(ui8flattenedWindow, ...
    dRmoonDEM, ...
    dSunDir_PixCoords, ...
    dAttMRP_fromTFtoCAM, ...
    dPosCam_TF, ...
    ui8coarseLimbPixels)%#codegen
arguments
    ui8flattenedWindow  (49,1)
    dRmoonDEM           (1,1)
    dSunDir_PixCoords   (2,1)
    dAttMRP_fromTFtoCAM (3,1)
    dPosCam_TF          (3,1)
    ui8coarseLimbPixels (2,1)
end
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


inputDataSample = zeros(60, 1, 'single');

inputDataSample(1:49)  = single(ui8flattenedWindow);
inputDataSample(50)    = single(dRmoonDEM);
inputDataSample(51:52) = single(dSunDir_PixCoords);
inputDataSample(53:55) = single(dAttMRP_fromTFtoCAM); % Convert Attitude matrix to MRP parameters
inputDataSample(56:58) = single(dPosCam_TF);
inputDataSample(59:60) = single(ui8coarseLimbPixels);

% dataBufferSize = uint16(60*8);

if iscolumn(inputDataSample)
    dataMessage = transpose(inputDataSample);
else
    dataMessage = (inputDataSample);
end

dataBuffer = typecast(dataMessage, 'uint8');
dataLength = typecast(uint32(length(dataBuffer)), 'uint8');
dataBufferToWrite = [dataLength, dataBuffer];

end


