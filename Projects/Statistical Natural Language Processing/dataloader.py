import numpy as np
import os
import pickle
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader


class eyeDataset(Dataset):

    '''Characterizes a dataset for PyTorch'''

    def __init__(self, data):
        '''Initialization'''

        self.data = data

    def __len__(self):
        '''Denotes the total number of samples'''

        return self.data.shape[0]

    def __getitem__(self, index):
        '''Generates one sample of data'''

        # Load data and get label

        X = self.data[index, 1:]
        y = self.data[index, 0]

        return (X, y)
