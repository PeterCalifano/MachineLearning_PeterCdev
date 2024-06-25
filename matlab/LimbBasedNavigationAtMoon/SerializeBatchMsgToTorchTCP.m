function [dataLength, dataBufferToWrite] = SerializeBatchMsgToTorchTCP(i_strInputSamplesBatchStruct, ...
                                                                       i_ui32Nbatches, ...
                                                                       i_ui32InputSize)
arguments
    i_strInputSamplesBatchStruct (:,:) {isstruct}
    i_ui32Nbatches               (1,1) uint32
    i_ui32InputSize              (1,1) uint32
end
%% PROTOTYPE
% [dataLength, dataBufferToWrite] = SerializeBatchMsgToTorchTCP(i_strInputSamplesBatchStruct, ...
%                                                                        i_ui32Nbatches, ...
%                                                                        i_ui32InputSize)
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the function does
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% i_strInputSamplesBatchStruct (:,:) {isstruct}
% i_ui32Nbatches               (1,1) uint32
% i_ui32InputSize              (1,1) uint32
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% out1 [dim] description
% Name1                     []
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 17-06-2024        Pietro Califano         Adapted from script.
% 20-06-2024        Pietro Califano         Update for integration with LUMIO IP
% 24-06-2024        Pietro Califano         Upgrade to serialize batch of input samples; assignment is now
%                                           agnostic wrt the input data. Order has to be managed in the
%                                           generation of the input struct (same as order in input sample).
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

% NOTE: Fields must be in this order unless some other criterion is used to specify how to serialize the
% data and let the receiver know how to deserialize. There must be an agreement in any case. Otherwise one
% end has to send this information. No workaround for this.
% i_ui8flattenedWindow
% i_dSunDir_PixCoords
% i_dAttMRP_fromTFtoCAM
% i_ui8coarseLimbPixels

inputDataBatch = zeros(i_ui32InputSize, i_ui32Nbatches, 'single');

% Get fieldnames and field sizes
fieldNamesCell = fieldnames(i_strInputSamplesBatchStruct(1));
fieldSizes = zeros(length(fieldNamesCell), 1);

for idF = 1:length(fieldNamesCell)
    fieldSizes(idF) = length(i_strInputSamplesBatchStruct(1).(fieldNamesCell{idF}));
end

% Assignment loop
for idB = 1:i_ui32Nbatches

    % Temporary variables
    tmpSample = zeros(i_ui32InputSize, 1, 'single');
    ptrToInput = 1;

    % Extract and assign idB input sample
    for idF = 1:length(fieldNamesCell)-1

        % Store data into chunk of sample
        tmpSample(ptrToInput:ptrToInput+fieldSizes(idF)-1)  = single(i_strInputSamplesBatchStruct(idB).(fieldNamesCell{idF}));

        % Update index
        ptrToInput = ptrToInput + fieldSizes(idF);

    end

    % Assign last input using end as assert
    tmpSample(ptrToInput:end) = single(i_strInputSamplesBatchStruct(idB).(fieldNamesCell{end}));

    % Assign temporary sample into data batch
    inputDataBatch(:, idB) = tmpSample;
end

% Reshape data batch to column vector before casting
inputDataBatch = reshape(inputDataBatch, [], 1);

if iscolumn(inputDataBatch)
    dataMessage = transpose(inputDataBatch);
else
    dataMessage = (inputDataBatch);
end

% Cast and serialize message string
dataBuffer = typecast(dataMessage, 'uint8');
dataBuffer = [typecast(uint32(i_ui32Nbatches), 'uint8'), dataBuffer];

dataLength = typecast(uint32(length(dataBuffer)), 'uint8');
dataBufferToWrite = [dataLength, dataBuffer];

end


