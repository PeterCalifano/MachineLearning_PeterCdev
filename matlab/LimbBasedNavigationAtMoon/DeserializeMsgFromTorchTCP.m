function [dPredictedPixCorrection] = DeserializeMsgFromTorchTCP(recvBytes, recvDataBuffer, outputSize)%#codegen
arguments
    recvBytes      (1,1)
    recvDataBuffer (:,1)
    outputSize      (1,1) = 2
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

%fprintf('\nReceived data length: %d. \nData vector: ', recvBytes);

% Compute number of batches
inputSizeInBytes = 4*outputSize;
numOfBatches = length(recvDataBuffer)/inputSizeInBytes;

dPredictedPixCorrection = zeros(outputSize, numOfBatches);

ptrVector = 1:inputSizeInBytes;

for idB = 1:numOfBatches

    % Extract data from buffer
    dataBufferReceived = typecast(recvDataBuffer(ptrVector), 'single');
    %disp(dataBufferReceived);

    % Update vector of index pointers
    ptrVector = ptrVector + inputSizeInBytes;

    % Store output vector
    dPredictedPixCorrection(:, idB) = double(dataBufferReceived);

end

end
