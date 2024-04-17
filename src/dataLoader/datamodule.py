import json

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

import numpy as np
import os
from h5pickle import h5py
import os.path as osp

transform = T.Compose([T.FaceToEdge(), T.ToUndirected()])


class DatasetDPBase(Dataset):

    def __init__(self, dInfo, dataset_dir, split='test'):

        self.dataset_dir = osp.join(dataset_dir, split+'.h5')
        assert os.path.isfile(self.dataset_dir), '%s not exist' % dataset_dir
        with open(dataset_dir + '/meta.json', 'r') as f:
            metadata = json.load(f)
        self.file_handle = h5py.File(self.dataset_dir, "r")
        self.data_keys = tuple(metadata['field_names'])
        self.steps_trajectory = int(metadata['trajectory_length'])

        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {'z': self.z_dim, 'q': 0, 'q_0': self.q_dim, 'n': 9, 'f': dInfo['dataset']['external_force_dim'],
                     'g': 0}
        # self.dims = {'z': self.z_dim, 'q': self.q_dim, 'q_0': 0, 'n': 1, 'f': dInfo['dataset']['external_force_dim'], 'g': 0}
        self.dt = dInfo['dataset']['dt']

    @staticmethod
    def datas_to_graph(datas, num, step, metadata):

        face = torch.from_numpy(datas[metadata.index('cells')].T).long()
        node_type = torch.tensor(datas[metadata.index('node_type')]).long()
        mesh_crds = torch.from_numpy(datas[metadata.index('pos')]).float()
        velocity = torch.from_numpy(datas[metadata.index('velocity')]).float()
        pressure = torch.from_numpy(datas[metadata.index('pressure')]).float()

        x = torch.cat((velocity[0], pressure[0]), dim=-1)
        y = torch.cat((velocity[1], pressure[1]), dim=-1)

        g = Data(x=x, y=y, face=face, n=node_type, pos=mesh_crds[0], num=num, step=step)

        g_edges = transform(g)
        g_edges.edge_index = add_self_loops(g_edges.edge_index)[0]

        return g_edges
    
    def get_dataset(self):
        return self.dataset
    
    def get_loader(self, batch_size=32, shuffle=False, num_workers=1):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.datasets)


class DatasetDP(DatasetDPBase):
    def __init__(self, dInfo, dataset_dir, split='test', trajectory=None):
        super().__init__(dInfo, dataset_dir, split)
        self.load_dataset(trajectory)

    def load_dataset(self, trajectory=None):

        self.dataset = []

        keys = list(self.file_handle.keys())
        if trajectory is not None:
            if isinstance(trajectory, int):
                keys = keys[: min(trajectory, len(keys))]
            elif isinstance(trajectory, list):
                keys = [keys[i] for i in trajectory]
            else:
                raise ValueError('Trajectory value not int or list!')

        self.trajectories = {k: self.file_handle[k] for k in keys}

        for num, trajectory in self.trajectories.items():

            for step in range(self.steps_trajectory-1):

                datas = []

                for k in self.data_keys:
                    if k in ["velocity", 'pressure', "pos"]:
                        r = np.array((trajectory[k][step], trajectory[k][step+1]), dtype=np.float32)
                    else:
                        r = trajectory[k][step]
                        if k in ["node_type", "cells"]:
                            r = r.astype(np.int32)
                    datas.append(r)

                graph = DatasetDPBase.datas_to_graph(datas, num, step, self.data_keys)
                self.dataset.append(graph)

    def get_stats(self):
        total_tensor = None
        total_tensor_pos = None

        sample = self.dataset

        for data in sample:
            if total_tensor is not None:
                total_tensor = torch.cat((total_tensor, data.x), dim=0)
            else:
                total_tensor = data.x

            if total_tensor_pos is not None:
                total_tensor_pos = torch.cat((total_tensor_pos, data.pos), dim=0)
            else:
                total_tensor_pos = data.pos

        scaler_var = StandardScaler()
        scaler_var.fit(total_tensor)

        scaler_pos = StandardScaler()
        scaler_pos.fit(total_tensor_pos)

        return scaler_var, scaler_pos

    def describe_dataset(self):

        print('Dataset loaded!')
        print(f'    Number trajectories: {len(self.trajectories)}')
        print(f'    steps trajectory: {self.steps_trajectory}')
        print()

class DatasetDPRollout(DatasetDPBase):
    def __init__(self, dInfo, dataset_dir, split='test', trajectory=None):
        super().__init__(dInfo, dataset_dir, split)
        self.load_dataset(trajectory)

    def load_dataset(self, trajectory=None):

        if trajectory is None:
            raise ValueError('Provide a simulation to Rollout')

        self.dataset = []

        keys = list(self.file_handle.keys())

        keys = [keys[trajectory]]

        self.trajectories = {k: self.file_handle[k] for k in keys}

        for num, trajectory in self.trajectories.items():

            for step in range(self.steps_trajectory-1):

                datas = []

                for k in self.data_keys:
                    if k in ["velocity", 'pressure', "pos"]:
                        r = np.array((trajectory[k][step], trajectory[k][step+1]), dtype=np.float32)
                    else:
                        r = trajectory[k][step]
                        if k in ["node_type", "cells"]:
                            r = r.astype(np.int32)
                    datas.append(r)

                graph = DatasetDPBase.datas_to_graph(datas, num, step, self.data_keys)
                self.dataset.append(graph)

    def get_stats(self):
        total_tensor = None
        total_tensor_pos = None

        sample = self.dataset

        for data in sample:
            if total_tensor is not None:
                total_tensor = torch.cat((total_tensor, data.x), dim=0)
            else:
                total_tensor = data.x

            if total_tensor_pos is not None:
                total_tensor_pos = torch.cat((total_tensor_pos, data.pos), dim=0)
            else:
                total_tensor_pos = data.pos

        scaler_var = StandardScaler()
        scaler_var.fit(total_tensor)

        scaler_pos = StandardScaler()
        scaler_pos.fit(total_tensor_pos)

        return scaler_var, scaler_pos

    def describe_dataset(self):

        print('Dataset loaded!')
        print(f'    Number trajectories: {len(self.trajectories)}')
        print(f'    steps trajectory: {self.steps_trajectory}')
        print()

    
