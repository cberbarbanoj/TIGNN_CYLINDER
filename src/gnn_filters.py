"""model.py"""

import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean


# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            # if k != len(layer_vec) - 2: self.layers.append(nn.LeakyReLU())
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.edge_mlp = MLP([3 * self.dim_hidden + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        if u is not None:
            out = torch.cat([edge_attr, src, dest, u[batch]], dim=1)
        else:
            out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.node_mlp = MLP(
            [2 * self.dim_hidden + dims['f'] + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src, dest = edge_index
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))
        # out = torch.cat([out, scatter_add(edge_attr, dest, dim=0, dim_size=x.size(0))], dim=1)
        if f is not None:
            out = torch.cat([x, out, f], dim=1)
        elif u is not None:
            out = torch.cat([x, out, u[batch]], dim=1)
        else:
            out = torch.cat([x, out, ], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[src], x[dest], edge_attr, u,
                                    batch if batch is None else batch[src])
        x = self.node_model(x, edge_index, edge_attr, f, u, batch)

        return x, edge_attr


class PlasticityGNN(pl.LightningModule):
    def __init__(self, dims, scaler, dInfo, save_folder, rollout_simulation=1, rollout_variable=None, rollout_freq=2):
        super().__init__()
        n_hidden = dInfo['model']['n_hidden']
        dim_hidden = dInfo['model']['dim_hidden']
        self.passes = dInfo['model']['passes']
        self.filters = dInfo['model']['filters']
        self.data_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        self.state_variables = dInfo['dataset']['state_variables']
        self.radius_connectivity = dInfo['dataset']['radius_connectivity']
        self.trainable_idx = np.arange(len( self.state_variables)).tolist()
        self.save_folder = save_folder

        # Encoder MLPs
        # self.encoder_node = MLP([dim_node] + [dim_hidden])
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        # self.encoder_edge = MLP([dim_edge] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_f = MLP([2] + n_hidden * [dim_hidden] + [1])

        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(self.filters):
            node_model = NodeModel(n_hidden, dim_hidden, self.dims)
            edge_model = EdgeModel(n_hidden, dim_hidden, self.dims)
            GraphNet = \
                MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)
        # self.processorNrm = nn.ModuleList()
        # for _ in range(passes):
        #     layer_norm = nn.LayerNorm(dim_hidden)
        #     self.processorNrm.append(layer_norm)
        # Decoder MLPs
        # self.decoder_E = MLP([dim_hidden] + n_hidden * [dim_hidden] + [1])
        self.decoder_E = MLP([dim_hidden * self.filters] + n_hidden * [dim_hidden] + [self.dim_z])
        # self.decoder_S = MLP([dim_hidden] + n_hidden * [dim_hidden] + [1])
        self.decoder_S = MLP([dim_hidden * self.filters] + n_hidden * [dim_hidden] + [self.dim_z])

        self.decoder_L = MLP([dim_hidden * self.filters] + n_hidden * [dim_hidden] + [
            int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP([dim_hidden * self.filters] + n_hidden * [dim_hidden] + [
            int(self.dim_z * (self.dim_z + 1) / 2)])

        diag = torch.eye(self.dim_z, self.dim_z)
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)
        self.scaler = scaler
        self.dt = dInfo['dataset']['dt']
        self.noise_var = dInfo['model']['noise_var']
        self.lambda_d = dInfo['model']['lambda_d']
        self.lr = dInfo['model']['lr']
        self.miles = dInfo['model']['miles']
        self.gamma = dInfo['model']['gamma']

        # Rollout simulation
        self.rollout_simulation = rollout_simulation
        self.rollout_variable = rollout_variable
        self.rollout_freq = dInfo['model']['rollout_freq']
        self.error_message_pass = []

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.bmm(L, dEdz) + torch.bmm(M, dSdz)
        deg_E = torch.bmm(M, dEdz)
        deg_S = torch.bmm(L, dSdz)

        return dzdt[:, :, 0], deg_E[:, :, 0], deg_S[:, :, 0]

    def pass_thought_net(self, z_t0, z_t1, edge_index, n, f, g=None, batch=None, val=False):

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)

        noise = (self.noise_var) * torch.randn_like(z_norm[n == 1])
        z_norm[n == 1] = z_norm[n == 1] + noise

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]

        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        f = self.encoder_f(torch.cat((f, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1))

        '''Process'''
        x_outs = []
        for GraphNet in self.processor:
            x_pos = x.clone()
            edge_attr_pos = edge_attr.clone()
            for i in range(self.passes):
                x_res_pos, edge_attr_res_pos = GraphNet(x_pos, edge_index, edge_attr_pos, f=f, u=g,
                                                             batch=batch)

                x_pos += x_res_pos
                edge_attr_pos += edge_attr_res_pos

                if self.current_epoch % 10 == 0 and val:
                    # store data fro visuals error message passing
                    self.error_message_pass.append(
                        [i, 0.5 * i / self.passes + self.current_epoch, float(x_res_pos.mean())])
                        # [i, 0.5 * i / self.passes + self.current_epoch, float(x.mean())])

            x_outs.append(x_pos.clone())

        x = torch.cat(x_outs, dim=1)
        '''Decode'''
        # Gradients
        dEdz = self.decoder_E(x)
        dSdz = self.decoder_S(x)
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)
        #
        '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L, 1, 2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M, torch.transpose(M, 1, 2))

        dzdt_net, deg_E, deg_S = self.integrator(L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2))

        dzdt = (z1_norm - z_norm) / self.dt

        return dzdt_net, deg_E, deg_S, dzdt, L, M
    def training_step(self, batch, batch_idx, g=None):

        # Extract data from DataGeometric
        if self.dims['f'] == 1:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        else:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, None

        dzdt_net, deg_E, deg_S, dzdt, L, M, z_passes = self.pass_thought_net( z_t0, z_t1, edge_index, n, f, g=g, batch=batch.batch)

        # Compute losses
        loss_z = torch.nn.functional.mse_loss(dzdt_net, dzdt)
        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        # loss = nn.functional.mse_loss(dzdt_net, dzdt)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.state_variables is not None:
            for i, variable in enumerate(self.state_variables):
                loss_variable = nn.functional.mse_loss(dzdt_net[:, i], dzdt[:, i])
                self.log(f"train_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, g=None):

        # Extract data from DataGeometric
        if self.dims['f'] == 1:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        else:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, None

        dzdt_net, deg_E, deg_S, dzdt, L, M, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, g=g, batch=batch.batch, val=True)

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)

        # Compute losses
        loss_z = torch.nn.functional.mse_loss(dzdt_net, dzdt)
        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        if self.state_variables is not None:
            for i, variable in enumerate(self.state_variables):
                loss_variable = nn.functional.mse_loss(dzdt_net[:, i], dzdt[:, i])
                self.log(f"val_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)

        if (self.current_epoch % self.rollout_freq == 0) and (self.current_epoch > 0):
            # if self.rollout_simulation in batch.idx:
            if len(self.rollouts_z_t1_pred) == 0:
                # Initial state
                self.rollouts_z_t1_pred.append(z_t0)
                self.rollouts_z_t1_gt.append(z_t0)
                # self.rollouts_idx.append(batch.idx)
                self.rollouts_idx.append(self.local_rank)

            # set only the predicted state variables
            z1_net = z_norm + self.dt * dzdt_net
            z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
                self.device)
            # z_t1_pred = torch.clone(z_t1)
            z_t1_pred = z1_net_denorm


            # append variables
            self.rollouts_z_t1_pred.append(z_t1_pred)
            self.rollouts_z_t1_gt.append(z_t1)
            self.rollouts_idx.append(self.local_rank)

    def predict_step(self, batch, batch_idx, g=None):

        # Extract data from DataGeometric
        if self.dims['f'] == 1:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        else:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, None

        dzdt_net, deg_E, deg_S, dzdt, L, M, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, g=g, batch=batch.batch)

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_net = z_norm + self.dt * dzdt_net

        z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
            self.device)

        return z1_net_denorm, z_t1, L, M

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True),
        #                 'monitor': 'val_loss'}
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

