'''
Script to demonstrate how to use LSTM for sequence modeling task using PyTorch.
Created by PeterC on 2024-07-19, from PyTorch tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as torchFcn
import torch.optim as optim

torch.manual_seed(1)


def RunLSTMmodelExample():
    # Define LSTM layer with
    lstm = nn.LSTM(input_size=3, hidden_size=3)  
    # NOTE: Input size is the size of the input vector at one instant of the sequence
    #       Hidden size is the size of the hidden state vector. 
    #       LSTM module can directly stack multiple layers one after the other, by specifying the third input
    

    inputs = [torch.randn(1, 3)
              for _ in range(5)]  # make a sequence of length 5


    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)


def RunSpeechTaggingWithLSTM():
    print("Running LSTM model example")


if __name__ == "__main__":
    print('-------------------- TUTORIAL: Sequence Models and LSTM ----------------------')
    print('LSTM use example:')
    RunLSTMmodelExample()

    print('LSTM for Part-of-Speech Tagging example:')
    RunSpeechTaggingWithLSTM()
    print('-------------------- END ----------------------')



