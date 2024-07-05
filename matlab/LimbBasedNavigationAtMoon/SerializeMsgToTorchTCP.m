function [dataLength, dataBufferToWrite] = SerializeMsgToTorchTCP(i_ui8flattenedWindow, ...
    i_dSunDir_PixCoords, ...
    i_dAttMRP_fromTFtoCAM, ...
    i_ui8coarseLimbPixels)%#codegen
arguments
    i_ui8flattenedWindow  (49,1)  uint8
    i_dSunDir_PixCoords   (2,1)   double
    i_dAttMRP_fromTFtoCAM (3,1)   double
    i_ui8coarseLimbPixels (2,1)   uint8
end
%% PROTOTYPE
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the function does
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% in1 [dim] description
% Name1                     []
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% out1 [dim] description
% Name1                     []
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 17-06-2024        Pietro Califano         Adapted from script.
% 20-06-2024        Pietro Califano         Update for integration with LUMIO IP
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code
% ui8flattenedWindow
% dRmoonDEM --> REMOVED
% dSunDir_PixCoords
% dAttDCM_fromTFtoCAM
% dPosCam_TF --> REMOVED
% ui8coarseLimbPixels

inputDataSample = zeros(56, 1, 'single');
ptrToInput = 1;

flattenedWindSize = length(i_ui8flattenedWindow);

inputDataSample(ptrToInput:ptrToInput+flattenedWindSize-1)  = single(i_ui8flattenedWindow);

% Update index
ptrToInput = ptrToInput + flattenedWindSize; 

inputDataSample(ptrToInput:ptrToInput+length(i_dSunDir_PixCoords)-1) = single(i_dSunDir_PixCoords);

% Update index
ptrToInput = ptrToInput + length(i_dSunDir_PixCoords); % Update index


inputDataSample(ptrToInput : ptrToInput + length(i_dAttMRP_fromTFtoCAM)-1) = single(i_dAttMRP_fromTFtoCAM); % Convert Attitude matrix to MRP parameters

% Update index
ptrToInput = ptrToInput + length(i_dAttMRP_fromTFtoCAM); % Update index

inputDataSample(ptrToInput:end) = single(i_ui8coarseLimbPixels);


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


