import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print(train_data[0])
    count = 0
    for inst in train_data:
        count += len(inst)
    assert count == len(train_data)*3319
    # print('[Info] Loading dev data...')
    # dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
    print(len(test_data))

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types



# Define the parameters dictionary
params = {
    'data': 'data/data_bookorder/fold1/',
    'epoch': 100,
    'batch_size': 4,
    'd_model': 512,
    'd_rnn': 64,
    'd_inner_hid': 1024,  # Assuming this is what you meant by 'd_inner'
    'd_k': 512,
    'd_v': 512,
    'n_head': 4,
    'n_layers': 4,
    'dropout': 0.1,
    'lr': 1e-4,
    'smooth': 0.1,
    'log': 'log_type2_epoch3_batch_16.txt'
}

# Print the parameters for verification
for key, value in params.items():
    print(f'{key}={value}')

# You can access these parameters directly from the dictionary like this:
print(params['data'])
print(params['epoch'])

# If you want to use them as command-line arguments like argparse, you can access them as follows:
class Params:
    def __init__(self, params_dict):
        self.__dict__.update(params_dict)

opt = Params(params)

""" prepare dataloader """
trainloader, testloader, num_types = prepare_dataloader(opt)