"""dataLoader.py"""

import os
import numpy as np
import itertools
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer


class GraphDataset(Dataset):
    def __init__(self, dInfo, dset_dir, leghth=0, short=False):
        'Initialization'
        self.short = short
        self.dset_dir = dset_dir

        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {'z': self.z_dim, 'q': 0, 'q_0': self.q_dim, 'n': 4, 'f': dInfo['dataset']['external_force_dim'], 'g': 0}
        # self.dims = {'z': self.z_dim, 'q': self.q_dim, 'q_0': 0, 'n': 1, 'f': dInfo['dataset']['external_force_dim'], 'g': 0}

        self.samplingFactor = dInfo['dataset']['samplingFactor']
        self.dt = dInfo['dataset']['dt'] * self.samplingFactor
        self.data = []
        # self.data = torch.load(os.path.join(self.dset_dir))
        self.data = torch.load(dset_dir)
        if leghth != 0:
            self.data = self.data[:leghth]
        if short:
            # self.data = random.sample(self.data, int(len(self.data) / 2))
            self.data = self.data[: int(len(self.data) / 2)]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)

    def get_stats(self):
        total_tensor = None
        # sample = random.sample(self.data, k=round(len(self.data) * 0.3))
        sample = self.data
        for data in sample:
            if total_tensor is not None:
                total_tensor = torch.cat((total_tensor, data.x), dim=0)
            else:
                total_tensor = data.x

        scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler =PowerTransformer(method='yeo-johnson', standardize=False)
        # scaler = StandardScaler()

        scaler.fit(total_tensor)
        # apply transform
        # standardized = scaler.transform(total_tensor)

        return scaler


if __name__ == '__main__':
    pass
