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
% dRmoonDEM --> REMOVED
% dSunDir_PixCoords
% dAttDCM_fromTFtoCAM
% dPosCam_TF --> REMOVED
% ui8coarseLimbPixels

inputDataSample = zeros(56, 1, 'single');
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


