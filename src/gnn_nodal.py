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
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2:
                self.layers.append(nn.SiLU())
                # self.layers.append(nn.ReLU())
                # self.layers.append(nn.LeakyReLU())

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

        # out_mean = scatter_mean(out, edge, dim=0)
        # out = out_mean[edge, :]
        # for i in range(out_mean.shape[0]):
        #     indices = torch.where(edge == i)[0]
        #     out[indices[0], :] = out_mean[i]
        #     out[indices[1], :] = out_mean[i]

        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.node_mlp = MLP(
            [2 * self.dim_hidden + dims['f'] + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden]) #TODO change -1
        # [2 * self.dim_hidden + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

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
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        self.state_variables = dInfo['dataset']['state_variables']
        self.radius_connectivity = dInfo['dataset']['radius_connectivity']
        self.trainable_idx = np.arange(len(self.state_variables)).tolist()
        self.save_folder = save_folder


        self.z_normalizer = Normalizer(size=self.dims['z'], name='node_normalizer', device='cpu')

        # Encoder MLPs
        # self.encoder_node = MLP([dim_node] + [dim_hidden])
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        # self.encoder_edge = MLP([dim_edge] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])

        # Processor MLPs
        # self.processor = nn.ModuleList()
        # for _ in range(self.passes):
        #     node_model = NodeModel(n_hidden, dim_hidden, self.dims)
        #     edge_model = EdgeModel(n_hidden, dim_hidden, self.dims)
        #     GraphNet = \
        #         MetaLayer(node_model=node_model, edge_model=edge_model)
        #     self.processor.append(GraphNet)

        node_model = NodeModel(n_hidden, dim_hidden, self.dims)
        edge_model = EdgeModel(n_hidden, dim_hidden, self.dims)
        self.GraphNet = \
            MetaLayer(node_model=node_model, edge_model=edge_model)

        self.decoder_E_n = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_S_n = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        # self.decoder_E_e = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        # self.decoder_S_e = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        # self.decoder_F = MLP([1] + n_hidden * [10] + [self.dim_z])

        # self.decoder_L = MLP([dim_hidden] + n_hidden * [dim_hidden] + [
        #     int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        # self.decoder_M = MLP(
        #     [dim_hidden] + n_hidden * [dim_hidden] + [int(self.dim_z * (self.dim_z + 1) / 2)])
            # [dim_hidden] + n_hidden*3 * [dim_hidden] + [self.dim_z **2])

        # self.decoder_edgeL = MLP([dim_hidden] + n_hidden * [dim_hidden] +[
        #     int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        # self.decoder_edgeM = MLP([dim_hidden] + n_hidden * [dim_hidden] + [int(self.dim_z * (self.dim_z + 1) / 2)])

        self.decoder_L = MLP([dim_hidden*3] + n_hidden * [dim_hidden]*2 + [
            int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP(
            [dim_hidden*3] + n_hidden * [dim_hidden]*2 + [int(self.dim_z * (self.dim_z + 1) / 2)])

        diag = torch.eye(self.dim_z, self.dim_z)
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)
        self.scaler_var = scaler_var
        self.scaler_pos = scaler_pos
        self.dt = dInfo['dataset']['dt']
        self.dataset_type = dInfo['dataset']['type']
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
        self.criterion = torch.nn.functional.mse_loss
        # self.criterion = torch.nn.functional.huber_loss
        # self.criterion = torch.nn.functional.l1_loss

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.matmul(L, dEdz) + torch.matmul(M, dSdz) #+ f
        # dzdt = torch.sparse.mm(L, dEdz) + torch.sparse.mm(M, dSdz)
        deg_E = torch.matmul(M, dEdz)
        deg_S = torch.matmul(L, dSdz)

        return dzdt, deg_E, deg_S

    def decoder(self, x, edge_attr, f, batch, src, dest, mask):
        '''Decode'''

        # Gradients
        dEdz = self.decoder_E_n(x).unsqueeze(-1)
        dSdz = self.decoder_S_n(x).unsqueeze(-1)
        # dEdz_tot = dEdz #+ scatter_add(dEdz_e, dest, dim=0)
        # dSdz_tot = dSdz #+ scatter_add(dSdz_e, dest, dim=0)

        l = self.decoder_L(torch.cat([edge_attr, x[src], x[dest]], dim=1))
        m = self.decoder_M(torch.cat([edge_attr, x[src], x[dest]], dim=1))


        L = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=l.device, dtype=l.dtype)
        M = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=m.device, dtype=m.dtype)
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m


        Ledges = torch.subtract(L, torch.transpose(L, 1, 2))
        Medges = torch.bmm(M, torch.transpose(M, 1, 2))/torch.max(M) #forzamos que la M sea SDP

        edges_diag = dest == src
        edges_neigh = src != dest

        L_dEdz = torch.matmul(Ledges, dEdz[dest, :, :])
        M_dSdz = torch.matmul(Medges, dSdz[dest, :, :])
        tot = (torch.matmul(Ledges[edges_diag, :, :],  dEdz) + torch.matmul(Medges[edges_diag, :, :],  dSdz))
        M_dEdz_L_dSdz = L_dEdz + M_dSdz

        dzdt_net = tot[:, :, 0] - scatter_add(M_dEdz_L_dSdz[:,:,0][edges_neigh,:], src[edges_neigh], dim=0)
        loss_deg_E = (torch.matmul(Medges[edges_diag, :, :],  dEdz)[:, :, 0] ** 2)[mask].mean()
        loss_deg_S = (torch.matmul(Ledges[edges_diag, :, :],  dSdz)[:, :, 0] ** 2)[mask].mean()

        return dzdt_net, loss_deg_E, loss_deg_S

    def pass_thought_net(self, z_t0, z_t1, edge_index, n, q_0=None, f=None, g=None, batch=None, val=False, passes_flag=False,
                         mode='val'):
        self.batch_size = torch.max(batch) + 1
        z_norm = torch.from_numpy(self.scaler_var.transform(z_t0.cpu())).float().to(self.device)
        # z_norm = self.z_normalizer(z_t0, mode)
        z1_norm = torch.from_numpy(self.scaler_var.transform(z_t1.cpu())).float().to(self.device)
        # z1_norm = self.z_normalizer(z_t1, mode)
        # n = n.to(self.device)
        # f = f.to(self.device)

        mask = torch.logical_or(n[:, 0] == 0, n[:, 0] == 5)
        if mode == 'train':
            # noise = (self.noise_var) * torch.randn_like(z_norm)
            if self.dims['n'] == 1:
                noise = (self.noise_var) * torch.randn_like(z_norm[mask])
                z_norm[mask] = z_norm[mask] + noise
            else:
                noise = (self.noise_var) * torch.randn_like(z_norm[n[:, 0] == 1])
                z_norm[n[:, 0] == 1] = z_norm[n[:, 0] == 1] + noise
            # z_norm = z_norm + noise

        # Eulerian simulation
        if q_0 is not None:
            q = torch.from_numpy(self.scaler_pos.transform(q_0.cpu())).float().to(self.device)
            v = z_norm
        # Lagrangian simulation
        else:
            q = z_norm[:, :self.dim_q]
            v = z_norm[:, self.dim_q:]

        if self.dims['n'] == 1:
            x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        else:
            x = torch.cat((v, n.type(torch.float32)), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        x_res_list = [[torch.clone(x), f]]

        # for GraphNet in self.processor:
        for i in range(self.passes):
            x_res, edge_attr_res = self.GraphNet(x, edge_index, edge_attr, f=f, u=g, batch=batch)
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
        if self.dataset_type=='fluid':

            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

            n_vaso = x[n == 0].shape[0]
            mask_fluid = ((edge_index >= n_vaso)[0, :]) & ((edge_index >= n_vaso)[1, :])
            edge_index = edge_index[:, mask_fluid]
            edge_index = edge_index - torch.min(edge_index)
            edge_attr = edge_attr[mask_fluid, :]
            x = x[n == 1]
            batch = batch[n == 1]

        else:
            edge_attr = edge_attr
            src = src
            dest = dest
        # dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x, edge_attr, f, batch, edge_index[0, :], edge_index[1, :])
        dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x, edge_attr, f, batch, src, dest, mask)

        dzdt = (z1_norm - z_norm) / self.dt

        if self.dataset_type == 'fluid':
            #Cojemos las particulas del vaso de gt y no las predecimos
            dzdt_net_b = dzdt.clone()
            dzdt_net_b[n == 1] = dzdt_net
            dzdt = dzdt[n == 1]
        else:
            dzdt_net_b = dzdt_net.reshape(dzdt.shape)

        loss_z = self.criterion(dzdt_net[mask], dzdt[mask])

        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)
        # loss = loss_z

        if mode != 'eval':
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            if mode == 'val':
                self.log(f"{mode}_deg_E", loss_deg_E, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_deg_S", loss_deg_S, prog_bar=False, on_step=False, on_epoch=True)

            if self.state_variables is not None:
                for i, variable in enumerate(self.state_variables):
                    loss_variable = self.criterion(dzdt_net.reshape(dzdt.shape)[:, i], dzdt[:, i])
                    self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)

        z_passes = []
        if passes_flag:
            for i, elements in enumerate(x_res_list):
                element = elements[0]

                # Build z_hat
                dz_t1_dt_hat, _, _ = self.decoder(element, edge_attr, f, batch, src, dest, mask)

                z_t1_hat_current = dz_t1_dt_hat * self.dt + z_t0
                if i == 0:
                    z_t1_hat_prev = z_t1_hat_current
                z_passes.append([i, torch.clone(z_t1_hat_current), elements[1], float(nn.functional.mse_loss(z_t1_hat_prev, z_t1_hat_current))])
                z_t1_hat_prev = z_t1_hat_current
        torch.cuda.empty_cache()
        return dzdt_net_b, loss, z_passes

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


        dzdt_net, loss, _ = self.extrac_pass(batch, 'val')

        z_norm = torch.from_numpy(self.scaler_var.transform(batch.x.cpu())).float().to(self.device)
        # z_norm = self.z_normalizer(z_t0, 'val')

        if (self.current_epoch % self.rollout_freq == 0) and (self.current_epoch > 0):
            # if self.rollout_simulation in batch.idx:
            if len(self.rollouts_z_t1_pred) == 0:
                # Initial state
                self.rollouts_z_t1_pred.append(batch.x)
                self.rollouts_z_t1_gt.append(batch.x)
                self.rollouts_idx.append(self.local_rank)

            # set only the predicted state variables
            z1_net = z_norm + self.dt * dzdt_net
            z1_net_denorm = torch.from_numpy(self.scaler_var.inverse_transform(z1_net.detach().to('cpu'))).float().to(
                self.device)
            # z1_net_denorm = self.z_normalizer.inverse(z1_net)
            # z_t1_pred = torch.clone(z_t1)
            z_t1_pred = z1_net_denorm

            # append variables
            self.rollouts_z_t1_pred.append(z_t1_pred)
            self.rollouts_z_t1_gt.append(batch.y)
            self.rollouts_idx.append(self.local_rank)

    def predict_step(self, batch, batch_idx, g=None, passes_flag=False):

        dzdt_net, loss, z_passes = self.extrac_pass(batch, 'eval', passes_flag=passes_flag)

        z_norm = torch.from_numpy(self.scaler_var.transform( batch.x.cpu())).float().to(self.device)
        # z_norm = self.z_normalizer(z_t0, 'val')
        z1_net = z_norm + self.dt * dzdt_net

        z1_net_denorm = torch.from_numpy(self.scaler_var.inverse_transform(z1_net.detach().to('cpu'))).float().to(
            self.device)
        # z1_net_denorm = self.z_normalizer.inverse(z1_net)

        return z1_net_denorm, batch.y, z_passes

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