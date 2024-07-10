function [dPredictedOutput] = callModelTorchTCP(i_strInputSamplesBatchStruct, ...
    i_ui32Nbatches, ...
    i_ui32InputSize, ... 
    i_ui32outputSize, ...
    commHandler)
arguments
    i_strInputSamplesBatchStruct  {isstruct}
    i_ui32Nbatches               (1,1) uint32
    i_ui32InputSize              (1,1) uint32
    i_ui32outputSize             (1,1) uint32
    commHandler                  (1,1)
end
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
% i_strInputSamplesBatchStruct (1,1) {isstruct}
% i_ui32Nbatches               (1,1) uint32
% i_ui32InputSize              (1,1) uint32
% i_ui32outputSize             (1,1) uint32
% commHandler                  (1,1)
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% dPredictedOutput   (2,1)  
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 17-06-2024        Pietro Califano         Function 1st version.
% 20-06-2024        Pietro Califano         Updated version with reduced input size.
% 25-06-2024        Pietro Califano         Modified to make function agnostic to input data.
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code
% Serialize message to send with input sample for Neural Network loaded in TorchTCP

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
