import os
import torch
import numpy as np
from src.utils.utils import print_error, generate_folder
from src.utils.plots import plot_2D_image, plot_2D, plot_image3D, plotError, plot_3D, video_plot_3D
from amb.metrics import rrmse_inf
from src.utils.utils import compute_connectivity


def compute_error(z_net, z_gt, state_variables):
    # Compute error
    e = z_net.numpy() - z_gt.numpy()
    gt = z_gt.numpy()

    error = {clave: [] for clave in state_variables}
    L2_list = {clave: [] for clave in state_variables}

    for i, sv in enumerate(state_variables):
        e_ing = rrmse_inf(z_gt[:, :, i:i+1], z_net[:, :, i:i+1])
        L2 = ((e[1:, :, i] ** 2).sum(1) / (gt[1:, :, i] ** 2).sum(1)) ** 0.5
        error[sv] = e_ing
        L2_list[sv].extend(L2)
    # plotError_2D(gt, z_net, L2_q, L2_v, L2_e, dEdt, dSdt, self.output_dir_exp)
    return error, L2_list

def roll_out(plasticity_gnn, dataloader, device, radius_connectivity, dim_data):
    data = [sample for sample in dataloader]
    # data = data[:65]
    dim_z = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]
    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt = torch.zeros(len(data) + 1, N_nodes, dim_z)

    # Initial conditions
    z_net[0] = data[0].x
    z_gt[0] = data[0].x

    z_denorm = data[0].x
    edge_index = data[0].edge_index

    if hasattr(data[0], 'q_0'):
        q_0 = data[0].q_0
    elif hasattr(data[0], 'pos'):
        q_0 = data[0].pos
    else:
        q_0 = None

    # for sample in data:
    try:
        for t, snap in enumerate(data):
            snap.x = z_denorm
            snap.edge_index = edge_index
            snap = snap.to(device)
            with torch.no_grad():
                z_denorm, z_t1, z_passes = plasticity_gnn.predict_step(snap, 1)

            pos = z_denorm[:, :3].clone()
            if dim_data == 2:
                pos[:, 2] = pos[:, 2] * 0

            # edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(device)  #TODO cambiar si es beam a TRUE
            edge_index = snap.edge_index

            z_net[t + 1] = z_denorm
            z_gt[t + 1] = z_t1
    except:
        print(f'Ha fallado el rollout en el momento: {t}')

    return z_net, z_gt, q_0
def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):

    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    dim_data = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
    # Make roll out
    import time
    start_time = time.time()
    z_net, z_gt, q_0 = roll_out(plasticity_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'], dim_data)
    print(f'El tiempo tardado en el rollout: {time.time()-start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net, z_gt, dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['dataset']['dataset_dim'] == '2D':
        plot_2D_image(z_net, z_gt, -1, 0, q_0=q_0, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=0, q_0=q_0)
    else:
        data = [sample for sample in test_dataloader]
        video_plot_3D(z_net, z_gt, save_dir=save_dir_gif,)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=-1, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=2, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=1, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=4, step=70, n=data[0].n)
        plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)

    # output = trainer.predict(model=plasticity_gnn, dataloaders=test_dataloader)
    #
    # z_net = torch.zeros(len(output), output[0][0].shape[0], output[0][0].shape[1])
    # z_gt = torch.zeros(len(output), output[0][0].shape[0], output[0][0].shape[1])
    #
    # for i, out in enumerate(output):
    #     z_net[i] = out[0]
    #     z_gt[i] = out[1]
