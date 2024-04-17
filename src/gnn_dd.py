"""model.py"""
import time
import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean
from src.utils.normalization import Normalizer
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
from torchmetrics import MeanSquaredLogError
from torch.nn import Linear
import torch.nn.functional as F


# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec, layer_norm=True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2:
                self.layers.append(nn.ReLU())
                # self.layers.append(nn.SiLU())
                # self.layers.append(nn.LeakyReLU())
                if layer_norm:  # Apply LayerNorm except for the last layer
                    self.layers.append(nn.LayerNorm(layer_vec[k + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(nn.Module):

    def __init__(self, custom_func=None):
        super(EdgeModel, self).__init__()
        self.net = custom_func

    def forward(self, node_attr, edge_attr, edge_index):
        src, dest = edge_index
        edges_to_collect = []

        src_attr = node_attr[src]
        dest_attr = node_attr[dest]

        edges_to_collect.append(src_attr)
        edges_to_collect.append(dest_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return edge_attr_


# Node model
class NodeModel(nn.Module):

    def __init__(self, custom_func=None):
        super(NodeModel, self).__init__()

        self.net = custom_func

    def forward(self, node_attr, edge_attr, edge_index):
        # Decompose graph
        edge_attr = edge_attr
        nodes_to_collect = []

        _, dest = edge_index
        num_nodes = node_attr.shape[0]
        agg_received_edges = scatter_add(edge_attr, dest, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        return x


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr):

        edge_attr = self.edge_model(x, edge_attr, edge_index)
        x = self.node_model(x, edge_attr, edge_index)

        return x, edge_attr


class PlasticityGNN(pl.LightningModule):
    def __init__(self, dims, scaler_var, scaler_pos, dInfo, save_folder, rollout_simulation=1, rollout_variable=None, rollout_freq=2):
        super().__init__()
        n_hidden = dInfo['model']['n_hidden']
        dim_hidden = dInfo['model']['dim_hidden']
        self.passes = dInfo['model']['passes']
        self.batch_size = dInfo['model']['batch_size']
        self.data_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']
        dim_node = self.dims['z'] - 1 + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        self.state_variables = dInfo['dataset']['state_variables']
        self.radius_connectivity = dInfo['dataset']['radius_connectivity']
        self.trainable_idx = np.arange(len(self.state_variables)).tolist()
        self.save_folder = save_folder

        # Normalizers
        self._node_normalizer = Normalizer(size=dim_node, name='node_normalizer', device=self.device)
        self._output_normalizer_acceleration = Normalizer(size=self.dim_z-1, name='output_normalizer_velocity', device=self.device)
        self._output_normalizer_pressure = Normalizer(size=1, name='output_normalizer_pressure', device=self.device)
        self._edge_normalizer = Normalizer(size=3, name='edge_normalizer', device=self.device)

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])

        # Graph-Bloks - Performs message passing
        self.nm_custom_func = MLP([2 * dim_hidden + self.dims['f'] + self.dims['g']] + n_hidden * [dim_hidden] + [dim_hidden])
        self.em_custom_func = MLP([3 * dim_hidden + self.dims['g']] + n_hidden * [dim_hidden] + [dim_hidden])
        node_model = NodeModel(self.nm_custom_func)
        edge_model = EdgeModel(self.em_custom_func)
        self.GraphNet = \
            MetaLayer(node_model=node_model, edge_model=edge_model)

        # Decoder
        self.decoder = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z], layer_norm=False)

        self.scaler_var = scaler_var
        self.scaler_pos = scaler_pos
        self.dt = dInfo['dataset']['dt']
        self.dataset_type = dInfo['dataset']['type']
        self.noise_var = dInfo['model']['noise_var']
        self.lr = dInfo['model']['lr']
        self.miles = dInfo['model']['miles']
        self.gamma = dInfo['model']['gamma']

        # Rollout simulation
        self.rollout_simulation = rollout_simulation
        self.rollout_variable = rollout_variable
        self.rollout_freq = dInfo['model']['rollout_freq']
        self.error_message_pass = []
        self.criterion = torch.nn.functional.mse_loss

    def update_node_attr(self, frames, n):
        node_feature = []

        node_feature.append(frames)
        node_type = n
        one_hot = torch.nn.functional.one_hot(node_type, 9).squeeze()
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr


    def pass_thought_net(self, z_t0, z_t1, edge_index, n, q_0=None, f=None, g=None, batch=None, val=False, passes_flag=False,
                         mode='val'):
        self.batch_size = torch.max(batch) + 1

        node_type = n
        frames = z_t0
        velocity = frames[:, :2]
        target = z_t1
        q = q_0

        mask_noise = node_type[:, 0] != 0
        mask_loss = torch.logical_or(node_type[:, 0] == 0, node_type[:, 0] == 5)
        if mode == 'train':
            noise = torch.normal(std=self.noise_var, mean=0.0, size=velocity.shape).to(self.device)
            noise[mask_noise] = 0
            velocity = velocity + noise

        node_attr = self.update_node_attr(velocity, node_type)
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)
        edge_attr = self._edge_normalizer(edge_attr, self.training)

        '''Encode'''
        x = self.encoder_node(node_attr)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        x_res_list = [[torch.clone(x), f]]

        # for GraphNet in self.processor:
        for i in range(self.passes):
            x_res, edge_attr_res = self.GraphNet(x, edge_index, edge_attr)
            # x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g, batch=batch)
            x += x_res
            edge_attr += edge_attr_res
            x_res_list.append([torch.clone(x_res), f])
            if self.current_epoch % 10 == 0 and val:
                # store data fro visuals error message passing
                self.error_message_pass.append(
                    [i, 0.5 * i / self.passes + self.current_epoch, float(x_res.mean())])
                # [i, 0.5 * i / self.passes + self.current_epoch, float(x.mean())])
            i += 1

        '''Decoder'''

        prediction = self.decoder(x)

        # acceleration = prediction[:, :2]
        # pressure = prediction[:, -1].unsqueeze(-1)
        #
        # predicted_velocity = velocity + acceleration
        # predicted_target = torch.cat((predicted_velocity, pressure), dim=1)

        # target_var = (target - frames) / self.dt
        acceleration_target = (target[:, :2] - velocity)
        pressure_target = target[:, -1].unsqueeze(-1)

        target_acceleration_norm = self._output_normalizer_acceleration(acceleration_target, self.training)
        target_pressure_norm = self._output_normalizer_pressure(pressure_target, self.training)

        target_norm = torch.cat((target_acceleration_norm, target_pressure_norm), dim=-1)

        loss = self.criterion(prediction[mask_loss], target_norm[mask_loss])

        if mode != 'eval':
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            if self.state_variables is not None:
                for i, variable in enumerate(self.state_variables):
                    loss_variable = self.criterion(prediction.reshape(target_norm.shape)[:, i][mask_loss], target_norm[:, i][mask_loss])
                    self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)

        z_passes = []
        if passes_flag:
            for i, elements in enumerate(x_res_list):
                element = elements[0]

                # Build z_hat
                dz_t1_dt_hat = self.decoder(element)

                z_t1_hat_current = dz_t1_dt_hat * self.dt + z_t0
                if i == 0:
                    z_t1_hat_prev = z_t1_hat_current
                z_passes.append([i, torch.clone(z_t1_hat_current), elements[1], float(nn.functional.mse_loss(z_t1_hat_prev, z_t1_hat_current))])
                z_t1_hat_prev = z_t1_hat_current
        torch.cuda.empty_cache()
        return prediction, loss, z_passes

    # def compute_loss(self, dzdt_net, dzdt, deg_E, deg_S, batch, mode='train'):
    #     # Compute losses
    #     loss_z = self.criterion(dzdt_net, dzdt)
    #     # loss_deg_E = (deg_E ** 2).mean()
    #     # loss_deg_S = (deg_S ** 2).mean()
    #
    #     loss_deg_E = torch.norm(deg_E)
    #     loss_deg_S = torch.norm(deg_S)
    #
    #     loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)  # /self.batch_size
    #
    #     # Logging to TensorBoard (if installed) by default
    #     self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    #     if mode == 'val':
    #         self.log(f"{mode}_deg_E", loss_deg_E, prog_bar=False, on_step=False, on_epoch=True)
    #         self.log(f"{mode}_deg_S", loss_deg_S, prog_bar=False, on_step=False, on_epoch=True)
    #
    #     if self.state_variables is not None:
    #         for i, variable in enumerate(self.state_variables):
    #             loss_variable = self.criterion(dzdt_net[:, i], dzdt[:, i])
    #             self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)
    #     return loss

    def extrac_pass(self, batch, mode, passes_flag=None):
        z_t0 = batch.x
        z_t1 = batch.y
        edge_index = batch.edge_index
        n = batch.n
        if 'q_0' in batch.keys():
            q_0 = batch.q_0
        elif 'pos' in batch.keys():
            q_0 = batch.pos
        else:
            q_0 = None
        f = batch.f if 'f' in batch.keys() else None
        g = batch.g if 'g' in batch.keys() else None

        # dzdt_net, deg_E, deg_S, dzdt, L, M, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, g=g, batch=batch.batch, passes_flag=passes_flag, mode='eval')
        dzdt_net, loss, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, q_0, f, g=None, batch=batch.batch,
                                                         passes_flag=passes_flag, mode=mode)

        return dzdt_net, loss, z_passes

    def training_step(self, batch, batch_idx, g=None):

        dzdt_net, loss, _ = self.extrac_pass(batch, 'train')

        return loss

    def validation_step(self, batch, batch_idx, g=None):

        z_0 = batch.x
        velocity_0 = z_0[:, :2]
        pressure_0 = z_0[:, -1].unsqueeze(-1)

        node_type = batch.n
        mask = torch.logical_or(node_type[:, 0] == 0, node_type[:, 0] == 5)

        z_1_net, loss, _ = self.extrac_pass(batch, 'val')
        acceleration_net = z_1_net[:, :2]
        pressure_net = z_1_net[:, -1].unsqueeze(-1)

        if (self.current_epoch % self.rollout_freq == 0) and (self.current_epoch > 0):
            # if self.rollout_simulation in batch.idx:
            if len(self.rollouts_z_t1_pred) == 0:
                # Initial state
                self.rollouts_z_t1_pred.append(batch.x)
                self.rollouts_z_t1_gt.append(batch.x)
                self.rollouts_idx.append(self.local_rank)

            # set only the predicted state variables
            # Denormalize acceleration and pressure
            acceleration_denorm = self._output_normalizer_acceleration.inverse(acceleration_net)
            pressure_denorm = self._output_normalizer_pressure.inverse(pressure_net)

            # Integrate acceleration to obtain velocity
            velocity_1 = velocity_0.clone()
            index_sum = mask.nonzero(as_tuple=True)
            velocity_1[index_sum] += acceleration_denorm[index_sum]

            pressure_1 = pressure_0.clone()
            pressure_1[index_sum] = pressure_denorm[index_sum]

            # Concatenate velocity and pressure
            z_1_pred = torch.cat((velocity_1, pressure_1), dim=-1)

            # append variables
            self.rollouts_z_t1_pred.append(z_1_pred)
            self.rollouts_z_t1_gt.append(batch.y)
            self.rollouts_idx.append(self.local_rank)

    def predict_step(self, batch, batch_idx, g=None, passes_flag=False):

        z_0 = batch.x
        velocity_0 = z_0[:, :2]
        pressure_0 = z_0[:, -1].unsqueeze(-1)

        node_type = batch.n
        mask = torch.logical_or(node_type[:, 0] == 0, node_type[:, 0] == 5)
        z_1_net, loss, z_passes = self.extrac_pass(batch, 'eval', passes_flag=passes_flag)
        # Extract acceleration and pressure
        acceleration_net = z_1_net[:, :2]
        pressure_net = z_1_net[:, -1].unsqueeze(-1)
        # Denormalize acceleration and pressure
        acceleration_denorm = self._output_normalizer_acceleration.inverse(acceleration_net)
        pressure_denorm = self._output_normalizer_pressure.inverse(pressure_net)
        # Integrate acceleration to obtain velocity
        velocity_1 = velocity_0.clone()
        index_sum = mask.nonzero(as_tuple=True)
        velocity_1[index_sum] += acceleration_denorm[index_sum]

        pressure_1 = pressure_0.clone()
        pressure_1[index_sum] = pressure_denorm[index_sum]
        # Concatenate velocity and pressure
        z_1_pred = torch.cat((velocity_1, pressure_1), dim=-1)

        return z_1_pred, batch.y, z_passes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, ),
        #     'monitor': 'train_loss'}
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True),
        #                 'monitor': 'val_loss'}
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}