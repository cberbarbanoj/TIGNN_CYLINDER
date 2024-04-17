"""model.py"""
import time
import torch
import torch.nn as nn
import numpy as np
from math import ceil
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean
from src.utils.normalization import Normalizer
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.pool import graclus
from torchmetrics import MeanSquaredLogError
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

# Multi Layer Perceptron (MLP) class
class GNN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask))[0,:,:])

        return x


class DiffPool(torch.nn.Module):
    def __init__(self, dim_hidden):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * 9022)

        num_classes = 9
        self.gnn1_pool = GNN2(dim_hidden, 64, num_nodes)
        self.gnn1_embed = GNN2(dim_hidden, 64, dim_hidden)
        #
        # num_nodes = ceil(0.25 * num_nodes)
        # self.gnn2_pool = GNN2(64, 64, num_nodes)
        # self.gnn2_embed = GNN2(64, 64, 64, lin=False)
        #
        # self.gnn3_embed = GNN2(64, 64, 64, lin=False)
        #
        # self.lin1 = torch.nn.Linear(64, 64)
        # self.lin2 = torch.nn.Linear(64, dim_hidden)

    def forward(self, x, adj, mask=None):
        self.s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, self.s, mask)
        # x_1 = s_0.t() @ z_0
        # adj_1 = s_0.t() @ adj_0 @ s_0

        # s = self.gnn2_pool(x, adj)
        # x = self.gnn2_embed(x, adj)
        #
        # x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        #
        # x = self.gnn3_embed(x, adj)

        return x, adj

    def upsamplig(self, x_red):
        return torch.matmul(torch.pinverse(self.s).transpose(1, 2), x_red)

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

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module

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
    def __init__(self, dims, scaler, dInfo, save_folder, rollout_simulation=1, rollout_variable=None, rollout_freq=2):
        super().__init__()
        n_hidden = dInfo['model']['n_hidden']
        dim_hidden = dInfo['model']['dim_hidden']
        self.passes = dInfo['model']['passes']
        self.batch_size = dInfo['model']['batch_size']
        self.data_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']
        dim_node = self.dims['z'] + self.dims['n']  - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        self.state_variables = dInfo['dataset']['state_variables']
        self.radius_connectivity = dInfo['dataset']['radius_connectivity']
        self.trainable_idx = np.arange(len(self.state_variables)).tolist()
        self.save_folder = save_folder

        self.diffPool = DiffPool(dInfo['model']['dim_hidden'])

        self.z_normalizer = Normalizer(size=self.dims['z'], name='node_normalizer', device='cuda')

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
        self.decoder_E_e = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_S_e = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_F = MLP([1] + n_hidden * [10] + [self.dim_z])

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
        self.criterion = torch.nn.functional.mse_loss
        # self.criterion = torch.nn.functional.huber_loss
        # self.criterion = torch.nn.functional.l1_loss

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.matmul(L, dEdz) + torch.matmul(M, dSdz) #+ f
        # dzdt = torch.sparse.mm(L, dEdz) + torch.sparse.mm(M, dSdz)
        deg_E = torch.matmul(M, dEdz)
        deg_S = torch.matmul(L, dSdz)

        # deg_E = torch.matmul(M, torch.sparse.mm(torch.transpose(M, 1, 0), dEdz))
        # # deg_E = torch.matmul(M, deg_E_)
        # deg_S = torch.sparse.mm(L, dSdz)
        # M_dSdz = torch.sparse.mm(torch.transpose(M, 1, 0), dSdz)
        # dzdt = torch.sparse.mm(L, dEdz) + M_dSdz# torch.sparse.mm(M, M_dSdz)  # + f
        del  M, L

        return dzdt, deg_E, deg_S

    def decoder(self, x, edge_attr, f, batch, src, dest):
        '''Decode'''
        # incoming_connections = torch.zeros(x.size(0), dtype=torch.int64, device=self.device)
        # incoming_connections.index_add_(0, dest, torch.ones(dest.size(0), dtype=torch.int64, device=self.device))
        # x = torch.cat([x,(incoming_connections).unsqueeze(1)], dim=1)

        # Gradients
        dEdz = self.decoder_E_n(x)
        dSdz = self.decoder_S_n(x)

        dEdz_e = self.decoder_E_e(edge_attr)
        dSdz_e = self.decoder_S_e(edge_attr)

        dEdz_tot = dEdz# + scatter_add(dEdz_e, dest, dim=0)
        dSdz_tot = dSdz #+ scatter_add(dSdz_e, dest, dim=0)
        # n_edges = scatter_add(torch.ones(dest.shape[0], device=self.device), dest, dim=0)


        # dSdz_tot = dSdz[batch == batch_i, :] + dSdz_e[ini_n:cnt_edge_node, :]

        # f_dec = self.decoder_F(f)
        # GENERIC flattened matrices
        # l = self.decoder_L(x)
        # l = self.decoder_L(torch.cat([edge_attr, x[dest]*x[src]*10], dim=1))
        l = self.decoder_L(torch.cat([edge_attr,x[src-torch.min(src)], x[dest-torch.min(dest)]], dim=1)) #MALP
        # l = self.decoder_L(torch.cat([edge_attr, x[dest],x[src]], dim=1))
        # m = self.decoder_M(x)
        m = self.decoder_M(torch.cat([edge_attr, x[src-torch.min(src)], x[dest-torch.min(dest)]], dim=1))
        # m = self.decoder_M(torch.cat([edge_attr, x[src], x[dest]], dim=1))

        loss_deg_E = 0
        loss_deg_S = 0
        cnt_n_node = 0
        cnt_edge_node = 0
        dzdt_net_b = []
        for batch_i in range(self.batch_size):
            '''Select the info of one simulation'''
            x_batch = x[batch == batch_i]
            x_batch_size = x_batch.size(0)
            # f_dec_batch = f_dec[batch == batch_i]
            # n_edges_batch = torch.repeat_interleave(n_edges[batch ==batch_i], self.dim_z)

            cnt_n_node += x_batch_size
            ini_n = cnt_n_node - x_batch_size

            src_batch = src[(src >= ini_n) & (src < cnt_n_node)]
            dest_batch = dest[(dest >= ini_n) & (dest < cnt_n_node)]
            src_batch = torch.subtract(src_batch, torch.min(src_batch))
            dest_batch = torch.subtract(dest_batch, torch.min(dest_batch))

            edge_batch_size = src_batch.size(0)
            cnt_edge_node += edge_batch_size
            ini_n = cnt_edge_node - edge_batch_size
            l_batch = l[ini_n:cnt_edge_node, :]
            m_batch = m[ini_n:cnt_edge_node, :]

            ############################
            '''Reparametrization'''
            L = torch.zeros(edge_batch_size, self.dim_z, self.dim_z, device=l.device, dtype=l.dtype)
            M = torch.zeros(edge_batch_size, self.dim_z, self.dim_z, device=m.device, dtype=m.dtype)

            L[:, torch.tril(self.ones, -1) == 1] = l_batch
            M[:, torch.tril(self.ones) == 1] = m_batch#*100

            Ledges = torch.subtract(L, torch.transpose(L, 1, 2))
            Ledges[src_batch<dest_batch, :, :] = - torch.transpose(Ledges[src_batch < dest_batch, :, :], 1, 2)

            Medges = M + torch.transpose(M, 1, 2)
            # self.ones = torch.ones(self.N_dim, self.N_dim)
            # M symmetric and positive semi-definite
            # Medges = torch.mul(M[src_batch, :, :], M[dest_batch, :, :])
            # Medges = M[src_batch, :, :] * M[dest_batch, :, :] * m_edge_batch
            # Medges = torch.bmm(Medges, torch.transpose(Medges, 1, 2))

            ############################
            N_dim = x_batch_size * self.dim_z

            ############################
            idx_row = []
            idx_col = []
            for i in range(self.dim_z):
                for j in range(self.dim_z):
                    if idx_col == []:
                        idx_row = (src_batch * self.dim_z)
                        idx_col = (dest_batch * self.dim_z)
                    else:
                        idx_row = torch.cat([idx_row, (src_batch * self.dim_z) + i], dim=0)
                        idx_col = torch.cat([idx_col, (dest_batch * self.dim_z) + j], dim=0)

            # inicio = time.time()
            # L_big = torch.zeros(N_dim, N_dim, device=l.device, dtype=l.dtype)
            # M_big = torch.zeros(N_dim, N_dim, device=m.device, dtype=m.dtype)
            # L_big[idx_row, idx_col] = torch.transpose(torch.transpose(Ledges, 1, 0), 2, 1).reshape(-1)
            # M_big[idx_row, idx_col] = torch.transpose(torch.transpose(Medges, 1, 0), 2, 1).reshape(-1)
            # print(f"Tiempo de ejecución de 0: {time.time() - inicio} segundos")

            i = [idx_row.tolist(), idx_col.tolist()]
            Ledges = torch.transpose(torch.transpose(Ledges, 1, 0), 2, 1).reshape(-1)
            Medges = torch.transpose(torch.transpose(Medges, 1, 0), 2, 1).reshape(-1)
            L_big = torch.sparse_coo_tensor(i, Ledges, (N_dim, N_dim))
            M_big = torch.sparse_coo_tensor(i, Medges, (N_dim, N_dim))
            #
            # M_big = torch.sparse.mm(M_big, M_big.T)

            ############################
            # for i in range(self.dim_z):
            # for i in range(self.dim_z):
            #     for j in range(self.dim_z):
            #         L_big[(src_batch * self.dim_z) + i, (dest_batch * self.dim_z) + j] = Ledges[:, i, j]
            #         M_big[(src_batch * self.dim_z) + i, (dest_batch * self.dim_z) + j] = Medges[:, i, j]
            #
            # M_big = torch.matmul(M_big, torch.transpose(M_big, 1, 0))/torch.max(M_big) #forzamos que la M sea SDP
            # L_big = torch.matmul(L_big, torch.transpose(L_big, 1, 0)) #forzamos que la M sea SDP
            # torch.allclose(-L_big, L_big.T)


            ########################
            dzdt_net, deg_E, deg_S = self.integrator(L_big, M_big,
                                                     dEdz_tot[batch == batch_i, :].reshape(-1, 1),
                                                     dSdz_tot[batch == batch_i, :].reshape(-1, 1))
                                                     # f_dec_batch.reshape(-1, 1))
            del M_big, L_big, M, L, dEdz_tot, dSdz_tot
            # torch.cuda.empty_cache()
            dzdt_net = dzdt_net[:, 0] #/ n_edges_batch
            dzdt_net_b.append(dzdt_net.reshape((x_batch_size, self.dim_z)))
            # loss_deg_E += torch.norm(deg_E)
            loss_deg_E += (deg_E ** 2).mean()
            # loss_deg_S += torch.norm(deg_S)
            loss_deg_S += (deg_S ** 2).mean()

        dzdt_net = torch.cat(dzdt_net_b, dim=0)

        return dzdt_net, loss_deg_E, loss_deg_S

    def pass_thought_net(self, z_t0, z_t1, edge_index, n, f, n_edge, g=None, batch=None, val=False, passes_flag=False,
                         mode='val'):
        self.batch_size = torch.max(batch) + 1
        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        # z_norm = self.z_normalizer(z_t0, mode)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)
        # z1_norm = self.z_normalizer(z_t1, mode)
        # n = n.to(self.device)
        # f = f.to(self.device)

        if mode == 'train':
            # noise = (self.noise_var) * torch.randn_like(z_norm)
            noise = (self.noise_var) * torch.randn_like(z_norm[n == 1][:, :self.dim_q])
            z_norm[n == 1][:, :self.dim_q][:, :self.dim_q] = z_norm[n == 1][:, :self.dim_q] + noise
            noise = (self.noise_var) * torch.randn_like(z_norm[n == 2][:, :self.dim_q])
            z_norm[n == 2][:, :self.dim_q] = z_norm[n == 2][:, :self.dim_q] + noise
            # z_norm = z_norm + noise

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]

        # one_hot = torch.nn.functional.one_hot(n.to(torch.int64), 3)
        # x = torch.cat((v, one_hot), dim=1)
        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        # u0 = q0[src] - q0[dest]
        # u0_norm = torch.norm(u0, dim=1).reshape(-1, 1)
        # edge_attr = torch.cat((u, u_norm, u0, u0_norm), dim=1)
        # edge_attr = torch.cat((u, u_norm, n_edge.reshape(-1, 1)), dim=1)
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
        if n_edge:
            edge_attr = edge_attr[n_edge == 0]
            src = src[n_edge == 0]
            dest = dest[n_edge == 0]
        else:
            mask_a1 = torch.zeros_like(src, dtype=torch.bool)
            mask_a2 = torch.zeros_like(dest, dtype=torch.bool)

            for val in n.nonzero().squeeze():
                mask_a1 = mask_a1 | (src == val)
                mask_a2 = mask_a2 | (dest == val)

            # Utilizamos las máscaras para seleccionar los elementos de a1 y a2
            src = src[mask_a1 & mask_a2]
            dest = dest[mask_a1 & mask_a2]
            edge_attr = edge_attr[mask_a1 & mask_a2]
            x = x[n == 1]
            batch = batch[n == 1]
        dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x, edge_attr, f, batch, src, dest)
        # adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
        #                           sparse_sizes=(x.shape[0], x.shape[0]))
        # x, adj_matrix= self.diffPool(x, adj_matrix.to_dense())
        # dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x[0:,:], edge_attr, f, batch, src, dest)

        dzdt = ((z1_norm - z_norm) / self.dt)[n == 1]
        loss_z = self.criterion(dzdt_net, dzdt)

        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S) / self.batch_size
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
                dz_t1_dt_hat, _, _ = self.decoder(element, edge_attr[n_edge == 0], f, batch, src[n_edge == 0], dest[n_edge == 0])

                z_t1_hat_current = dz_t1_dt_hat * self.dt + z_t0
                if i == 0:
                    z_t1_hat_prev = z_t1_hat_current
                z_passes.append([i, torch.clone(z_t1_hat_current), elements[1], float(nn.functional.mse_loss(z_t1_hat_prev, z_t1_hat_current))])
                z_t1_hat_prev = z_t1_hat_current
        print(torch.cuda.memory_allocated() / 1024 / 1024)
        print(torch.cuda.memory_reserved() / 1024 / 1024)
        return dzdt_net.reshape(dzdt.shape), loss, z_passes

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
        # Extract data from DataGeometric
        if self.dims['f'] == 1:
            z_t0, z_t1, edge_index, n, f, n_edge = batch.x, batch.y, batch.edge_index, batch.n, batch.f, batch.n_edge
        else:
            z_t0, z_t1, edge_index, n, f, n_edge = batch.x, batch.y, batch.edge_index, batch.n, None, None

        # dzdt_net, deg_E, deg_S, dzdt, L, M, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, g=g, batch=batch.batch, passes_flag=passes_flag, mode='eval')
        dzdt_net, loss, z_passes = self.pass_thought_net(z_t0, z_t1, edge_index, n, f,  n_edge, g=None, batch=batch.batch,
                                                         passes_flag=passes_flag, mode=mode)
        return dzdt_net, loss, z_passes

    def training_step(self, batch, batch_idx, g=None):

        dzdt_net, loss, _ = self.extrac_pass(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx, g=None):


        dzdt_net, loss, _ = self.extrac_pass(batch, 'val')

        z_norm = torch.from_numpy(self.scaler.transform(batch.x.cpu())).float().to(self.device)
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
            z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
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

        z_norm = torch.from_numpy(self.scaler.transform( batch.x.cpu())).float().to(self.device)
        # z_norm = self.z_normalizer(z_t0, 'val')
        z1_net = z_norm + self.dt * dzdt_net

        z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
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
