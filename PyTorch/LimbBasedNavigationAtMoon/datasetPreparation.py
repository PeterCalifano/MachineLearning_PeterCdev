# Script to prepare datasets for training/validation/testing of CNN-NN for Limb based navigation enhancement
# Created by PeterC 31-05-2024

import sys, platform, os
import argparse
import csv 
import json

# Auxiliary modules
import numpy as np
import matplotlib as mpl

# Torch modules
import customTorchTools


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dataset preparation pipeline.')

    parser.add_argument('--loadMode',
                        type=str,
                        default='json',
                        help='Specify the data format of the datapairs to process.')
    
    parser.add_argument('--dataPath',
                        type=str,
                        default='.',
                        help='Path to JSON file containing datapairs to process.')

    parser.add_argument('--outputPath',
                        type=str,
                        default='dataset',
                        help='Path to folder where postprocessed dataset will be saved.')

    return parser.parse_args()


def LoadJSONdata(dataFilePath):

    if not (os.path.isfile(dataFilePath)):
        raise FileExistsError('Data file not found. Check specified dataPath.')
    
    print('Data file FOUND, loading...')
    with open(dataFilePath, 'r') as json_file:
        try:
            dataJSONdict = json.load(json_file) # Load JSON as dict
        except Exception as exceptInstance:
            raise Exception('ERROR occurred:', exceptInstance.args)
        print('Data file: LOADED.')

    if isinstance(dataJSONdict, dict):
        # Get JSONdict data keys
        dataKeys = dataJSONdict.keys()
        # Print JSON structure
        
        #print('Loaded JSON file structure:')
        #print(json.dumps(dataJSONdict, indent=2, default=str))

    elif isinstance(dataJSONdict, list):
        raise Exception('Decoded JSON as list not yet handled by this implementation. If JSON comes from MATLAB jsonencode(), make sure you are providing a struct() as input and not a cell.')
    else: 
        raise Exception('ERROR: incorrect JSON file formatting')
    
    return dataJSONdict, dataKeys
    
def main(args):
    print('No code to execute as main')
    if args.loadMode == 'json':
        dataDict, dataKeys = LoadJSONdata(args.dataPath)
        

# Script executed as main
if __name__ == '__main__':
    print('Executing script as main')
    args = parse_args()
    main(args)