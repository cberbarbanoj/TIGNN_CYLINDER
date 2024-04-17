"""plot.py"""

import os
import json
import torch
import numpy as np
import moviepy.editor as mp
import open3d as o3d
from PIL import Image
import cv2
from tqdm import tqdm
import plotly.graph_objs as go
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_2D_image(z_net, z_gt, step, var=0, q_0=None, output_dir='outputs'):
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)

    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))

    ax1.set_title('Thermodynamics-informed GNN')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error')
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    # Adjust ranges
    if q_0 is not None:
        X, Y = q_0[:, 0].numpy(), q_0[:, 1].numpy()
    else:
        X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[step, :, var].min(), z_gt[step, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    # Initial snapshot
    if q_0 is not None:
        q1_net, q3_net = q_0[:, 0], q_0[:, 1]
        q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
    else:
        q1_net, q3_net = z_net[step, :, 0], z_net[step, :, 1]
        q1_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 1]
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net, vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max, vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error, vmax=var_error_max, vmin=var_error_min)
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    fig.savefig(os.path.join(output_dir, f'beam_{step}.png'))

    # Oculta las marcas de los ejes y las etiquetas
    ax1.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(output_dir, f'beam.svg'), format="svg")


def plot_2D(z_net, z_gt, save_dir, var=0, q_0=None):
    T = z_net.size(0)
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Adjust ranges
    if q_0 is not None:
        X, Y = q_0[:, 0].numpy(), q_0[:, 1].numpy()
    else:
        X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())

    # Initial snapshot
    if q_0 is not None:
        q1_net, q3_net = q_0[:, 0], q_0[:, 1]
        q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
    else:
        q1_net, q3_net = z_net[0, :, 0], z_net[0, :, 1]
        q1_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
        # Bounding box
        for xb, yb in zip(Xb, Yb):
            ax1.plot([xb], [yb], 'w')
            ax2.plot([xb], [yb], 'w')
            ax3.plot([xb], [yb], 'w')
        # Scatter points

        if q_0 is not None:
            q1_net, q3_net = q_0[:, 0], q_0[:, 1]
            q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
        else:
            q1_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 1]
            q1_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 1]

        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net

        ax1.scatter(q1_net, q3_net, c=var_net, vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max, vmin=z_min)
        ax3.scatter(q1_net, q3_net, c=var_error, vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join('images/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.mp4')
    anim.save(save_dir, writer=writergif)
    plt.close('all')


def plot_3D(z_net, z_gt, save_dir, var=0):
    T = z_net.size(0)

    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 1, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y'), ax3.set_zlabel('Z')
    ax1.view_init(elev=0., azim=90)
    ax2.view_init(elev=0., azim=90)
    ax3.view_init(elev=0., azim=90)

    # Adjust ranges
    X, Y, Z = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy(), z_gt[:, :, 2].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[0, :, 0], z_net[0, :, 2], z_net[0, :, 1]
    q1_gt, q2_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 2], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [zb], [yb], 'w')
        ax2.plot([xb], [zb], [yb], 'w')
        ax3.plot([xb], [zb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q2_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q2_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q2_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y'), ax3.set_zlabel('Z')
        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [zb], [yb], 'w')
            ax2.plot([xb], [zb], [yb], 'w')
            ax3.plot([xb], [zb], [yb], 'w')
        # Scatter points
        q1_net, q2_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 2], z_net[snap, :, 1]
        q1_gt, q2_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 2], z_gt[snap, :, 1]
        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net
        ax1.scatter(q1_net, q2_net, q3_net, c=var_net,
                    vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q2_gt, q3_gt, c=var_gt, vmax=z_max,
                    vmin=z_min)
        ax3.scatter(q1_net, q2_net, q3_net, c=var_error,
                    vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join(r'/home/atierz/Documentos/code/Experiments/fase_3D/frames', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=8)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.gif')
    anim.save(save_dir, writer=writergif)


def generate_pointclud(z_net, n, name=''):
    # Crear la paleta de colores
    data = z_net[1, :, -1]  # [n == 1]
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(data))
    colors[n == 0, :] = np.array([0.8, 0.8, 0.8, 1])
    colores = o3d.utility.Vector3dVector(colors[:, :-1])

    # Crear la ventana de visualización
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    view_control = visualizer.get_view_control()

    for i in range(z_net.shape[0]):
        # Crear la nube de puntos
        pcd = o3d.geometry.PointCloud()

        xyz = z_net[i, :, 0:3]
        pcd.points = o3d.utility.Vector3dVector(xyz)  # [n == 1, :])
        pcd.colors = colores

        visualizer.add_geometry(pcd)

        # # Girar el punto de vista de la cámara alrededor del eje Y
        view_control.rotate(i * -0.0, 80)  # Ajusta el ángulo de rotación según tus necesidades
        view_control.set_zoom(0.8)
        # view_control.rotate(0, 30)

        # Actualizar la visualización y guardar el frame
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(f'images/{name}_frame_{i}.png', do_render=True)

        # Borrar la nube de puntos pcd1
        visualizer.clear_geometries()


def video_plot_3D(z_net, z_gt, save_dir, n=[]):
    generate_pointclud(z_gt, n, name='gt')
    generate_pointclud(z_net, n, name='net')
    image_lst = []
    for i in range(z_net.shape[0]):
        frame_gt = cv2.cvtColor(cv2.imread(f'images/gt_frame_{i}.png'), cv2.COLOR_BGR2RGB)
        frame_net = cv2.cvtColor(cv2.imread(f'images/net_frame_{i}.png'), cv2.COLOR_BGR2RGB)

        frame_gt = cv2.resize(frame_gt[100:-100, 450:-450, :], None, fx=0.8, fy=0.8)
        frame_net = cv2.resize(frame_net[100:-100, 450:-450, :], None, fx=0.8, fy=0.8)

        # Asegúrate de que ambas imágenes tengan la misma altura
        altura = min(frame_gt.shape[0], frame_net.shape[0])

        # Concatena las imágenes horizontalmente
        imagen_concatenada = np.concatenate((frame_gt[:altura, :], frame_net[:altura, :]), axis=1)
        image_lst.append(imagen_concatenada)

    imageio.mimsave(os.path.join(save_dir, 'video.gif'), image_lst, fps=20, loop=4)


def plot_image3D(z_net, z_gt, save_folder, var=0, step=-1, n=[]):
    fig = plt.figure(figsize=(24, 20))
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 1, projection='3d')
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plt.axis('off')
    ax1.set_title('Thermodynamics-informed GNN')
    ax2.set_title('Ground Truth')
    ax3.set_title('Thermodynamics-informed GNN error')
    ax1.view_init(elev=10., azim=90)
    ax2.view_init(elev=10., azim=90)
    ax3.view_init(elev=10., azim=90)
    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))
    # Adjust ranges
    X, Y, Z = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy(), z_gt[:, :, 2].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[step, :, 0], z_net[step, :, 2], z_net[step, :, 1]
    q1_gt, q2_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 2], z_gt[step, :, 1]
    # var_net = calculateBorders(z_net[-1, :, :3], h, r1, r2)
    # var_gt = calculateBorders(z_gt[-1, :, :3], h, r1, r2)
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [zb], [yb], 'w')
        ax2.plot([xb], [zb], [yb], 'w')
        ax3.plot([xb], [zb], [yb], 'w')
    # Scatter points
    # ax1.set(xlim=(-0.04, 0.04), ylim=(-0.04, 0.04), zlim=(-0.01, 0.08))
    glass_index = np.where(n == 0)
    fluid_index = np.where(n == 1)
    ax1.scatter(q1_net[glass_index], q2_net[glass_index], q3_net[glass_index], alpha=0.1)
    s1 = ax1.scatter(q1_net[fluid_index], q2_net[fluid_index], q3_net[fluid_index], alpha=0.8, c=var_gt[fluid_index],
                     vmax=z_max, vmin=z_min)
    ax2.scatter(q1_gt[glass_index], q2_gt[glass_index], q3_gt[glass_index], alpha=0.1)
    s2 = ax2.scatter(q1_gt[fluid_index], q2_gt[fluid_index], q3_gt[fluid_index], alpha=0.8, c=var_gt[fluid_index],
                     vmax=z_max, vmin=z_min)

    ax3.scatter(q1_net[glass_index], q2_net[glass_index], q3_net[glass_index], alpha=0.1)
    s3 = ax3.scatter(q1_net[fluid_index], q2_net[fluid_index], q3_net[fluid_index], alpha=0.8, c=var_error[fluid_index],
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)
    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    plt.savefig(os.path.join(save_folder, f"grafico{str(var)}.svg"), format="svg")


def plt_LM(L, save_dir):
    L = L[10, :, :].cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X'), ax.set_ylabel('Y')
    x_data, y_data = np.meshgrid(range(L.shape[0]), range(L.shape[1]), indexing='ij')
    scatter = ax.scatter(x_data, y_data, c=L.flatten(), cmap='viridis', marker='o')
    fig.colorbar(scatter, ax=ax, location='bottom', pad=0.08)
    plt.savefig(save_dir)


def generatePlot2(i_size, j_size, particleList, variableList, tensorData1, tensorData2, titleList=[], title=' '):
    fig = plt.figure(figsize=(15, 7))
    # plt.title(title)

    for i in range(i_size):
        for j in range(j_size):
            index = i * j_size + j
            ax1 = fig.add_subplot(i_size, j_size, index + 1)
            ax1.set_title(f'{title} particle: {str(j)}'), ax1.grid()
            ax1.plot(np.asarray(tensorData1[:, particleList[j], variableList[i]]), linestyle='dotted',
                     label=titleList[0])
            ax1.plot(np.asarray(tensorData2[:, particleList[j], variableList[i]]), label=titleList[1])
            ax1.legend()
            # if len(titleList):
            #     ax1.set_title(titleList[index]), ax1.grid()
            # else:



def plotError(gt, z_net, L2_list, state_variables, dataset_dim, output_dir_exp):
    n_nodes = gt.shape[1]

    fig = plt.figure(figsize=(20, 20))

    for i, name in enumerate(L2_list):
        ax1 = fig.add_subplot(len(state_variables), 2, i * 2 + 1)
        ax1.set_title(name), ax1.grid()
        ax1.plot((gt[:, :, i]).sum((1)), linestyle='dotted', color='purple', label=f'{name} GT')
        ax1.plot((z_net.numpy()[:, :, i]).sum((1)), color='purple', label=f'{name} net')
        ax1.legend()

        ax2 = fig.add_subplot(len(state_variables), 2, i * 2 + 2)
        ax2.set_title('Error L2'), ax2.grid()
        ax2.plot(L2_list[name], color='purple', label=f'{name} GT')
    plt.savefig(os.path.join(output_dir_exp, 'L2_error.png'))

    if dataset_dim == '2D' or dataset_dim == 2:
        generatePlot2(2, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [0], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='Velocity - X')
        plt.savefig(os.path.join(output_dir_exp, 'position_error.png'))
        generatePlot2(2, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [1], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='Velocity - Y')
        plt.savefig(os.path.join(output_dir_exp, 'velocity_error.png'))
        generatePlot2(3, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [2], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='Pressure')
        plt.savefig(os.path.join(output_dir_exp, 'ss_error.png'))

    else: #TODO hacer bien
        generatePlot2(3, 4, [-3000, -2000, -1000, -1], [0, 1, 2], gt, z_net.numpy(), titleList=['gt', 'predicted'],
                      title='Position')
        plt.savefig(os.path.join(output_dir_exp, 'position_error.png'))
        generatePlot2(3, 4, [-3000, -2000, -1000, -1], [3, 4, 5], gt, z_net.numpy(), titleList=['gt', 'predicted'],
                      title='Velocity')
        plt.savefig(os.path.join(output_dir_exp, 'velocity_error.png'))
        generatePlot2(1, 4, [-3000, -2000, -1000, -1], [6], gt, z_net.numpy(), titleList=['gt', 'predicted'],
                      title='Energy')
        plt.savefig(os.path.join(output_dir_exp, 'energy_error.png'))



def plot_graph_data_plotly(coord_x, coord_y, nodes_data, n=None, edge_index=None, u=None,
                           lim_values=(0, 2), lim_coord_x=(0, 1.), lim_coord_y=(0, 1.),
                           title=None, save_path=None, colorscale='Viridis'):
    """
    Create a graphical representation of a network graph with colored nodes and edges using Plotly.

    Parameters:
    - coord_x (list): List of x-coordinates for the nodes.
    - coord_y (list): List of y-coordinates for the nodes.
    - nodes_data (list): List of data associated with each node, which is used for node color.
    - edge_index (tuple of lists): Tuple containing two lists representing the edges in the graph.
    - title (str): A title for the graph (optional).

    This function takes node coordinates, node data, and edge information to generate a graphical representation
    of a network graph. It assigns colors to the nodes based on the provided data and displays the graph with
    labeled axes and an optional title.

    The color of the nodes is determined by the 'nodes_data' parameter, and the color scale used is 'Viridis.'
    The edges are drawn in gray.

    Example usage:
    plot_graph_data(coord_x, coord_y, nodes_data, edge_index, title='Graph Visualization')
    """

    # Create scatter plot for nodes
    if edge_index is not None:
        scatter_nodes = go.Scatter(
            x=coord_x,
            y=coord_y,
            text=[str(i) for i in np.arange(len(coord_x))],  # Convert node numbers to strings
            # mode='markers+text',
            mode='markers',
            textposition='bottom center',
            textfont=dict(color='black'),
            marker=dict(
                size=10,
                color=nodes_data,
                colorscale=colorscale,
                cmin=lim_values[0],
                cmax=lim_values[1],
                colorbar=dict(title='Node Data'),
            ),
        )

        # Set nodes coordinates
        scatter_edges = []
        pos = {i + 1: (coord_x[i], coord_y[i]) for i in range(len(coord_x))}
        for i in range(len(edge_index[0])):
            source = edge_index[0][i] + 1
            target = edge_index[1][i] + 1
            # Create scatter plot for edges
            scatter_edges.append(go.Scatter(
                x=[pos[source][0], pos[target][0]],
                y=[pos[source][1], pos[target][1]],
                mode='markers+lines',
                line=dict(color='gray', width=1),
                marker=dict(
                    symbol="arrow",
                    size=10,
                    color='black',
                    angleref="previous",
                ),
            ))

    else:
        scatter_nodes = go.Scatter(
            x=coord_x,
            y=coord_y,
            text=[str(i) for i in np.arange(len(coord_x))],  # Convert node numbers to strings
            # mode='markers+text',
            mode='markers',
            textposition='bottom center',
            textfont=dict(color='black'),
            marker=dict(
                size=10,
                color=nodes_data,
                colorscale=colorscale,
                cmin=lim_values[0],
                cmax=lim_values[1],
                colorbar=dict(title='Node Data'),
            ),
        )

    if n is not None:
        # Create scatter plot for Boundary conditions

        bc_idx = np.argwhere(n != 0)[0, :].reshape(-1)

        color_mapping = {
            1: 'black',
            2: 'purple',
            3: 'pink',
            4: 'blue',
            5: 'grey',
        }
        color = [color_mapping[int(val)] for val in n[bc_idx]]

        scatter_nodes_bc = go.Scatter(
            x=np.array(coord_x)[bc_idx],
            y=np.array(coord_y)[bc_idx],
            mode='markers',
            marker=dict(
                size=10,
                color='rgba(0,0,0,0)',
                line=dict(
                    color=color,
                    width=2,  # Adjust the width of the contour line
                ),
            ),
        )

    if u is not None:
        # Set nodes coordinates
        scatter_u = []
        u_coord = np.argwhere(np.array(u) != 0.).reshape(-1)
        for i in u_coord:
            # Create scatter plot for imposed Us
            scatter_u.append(go.Scatter(
                x=[coord_x[i], coord_x[i]],
                y=[coord_y[i], coord_y[i]+50*u[i]],
                text=[f'{np.round(u[i], decimals=4)}'],
                # mode='lines+markers+text',
                mode='lines+markers',
                line=dict(color='gray', width=1),
                marker=dict(
                    symbol="arrow",
                    size=10,
                    color='black',
                    angleref="previous",
                ),
                textposition='top center',
            ))

    # Create a layout for the plot
    layout = go.Layout(
        title=title,
        xaxis=dict(title='x-coordinate', range=lim_coord_x, scaleratio=1),
        yaxis=dict(title='y-coordinate', range=lim_coord_y,),
        showlegend=False,
        title_x=0.5,
    )

    # Combine scatter plots and layout
    if (edge_index is not None) and (u is not None) and (n is not None):
        fig = go.Figure(data=scatter_edges + [scatter_nodes] + [scatter_nodes_bc] + scatter_u, layout=layout)
    elif (u is not None) and (n is not None):
        fig = go.Figure(data=[scatter_nodes] + [scatter_nodes_bc] + scatter_u, layout=layout)
    elif (u is not None) and (edge_index is not None):
        fig = go.Figure(data=scatter_edges + [scatter_nodes] + scatter_u, layout=layout)
    elif (u is not None):
        fig = go.Figure(data=[scatter_nodes] + scatter_u, layout=layout)
    else:
        fig = go.Figure(data=[scatter_nodes], layout=layout)

    if save_path is not None:
        fig.write_image(save_path, width=1000, height=500, engine='kaleido', scale=2)
    else:
        fig.show()


def make_gif(data, path, title, plot_variable, state_variables, with_edges=False, colorscale='Viridis'):

    # if len(state_variables) != data[0].x.shape[1]:
    #     raise KeyError('Wrong STATE_VARIABLES provided for given data!')
    path.mkdir(exist_ok=True, parents=True)

    VAR = state_variables.index(plot_variable)

    if 'COORD.COOR1' in state_variables:
        COORDX, COORDY = state_variables.index('COORD.COOR1'), state_variables.index('COORD.COOR2')
    elif 'pos' in data[0].keys():
        COORDX, COORDY = data[0].pos[:, 0], data[0].pos[:, 1]
    else:
        COORDX, COORDY = data[0].q_0[:, 0], data[0].q_0[:, 1]

    min_value, max_value = np.round(min(float(data[0].x[:, VAR].min()), float(data[-1].x[:, VAR].min())), decimals=3), np.round(max(float(data[0].x[:, VAR].max()), float(data[-1].x[:, VAR].max()))*1.1, decimals=3)
    if 'COORD.COOR1' in state_variables:
        min_coord_x, max_coord_x = min(float(data[0].x[:, COORDX].min()), float(data[-1].x[:, COORDX].min())) - 0.2, max(float(data[0].x[:, COORDX].max()), float(data[-1].x[:, COORDX].max()))*1.1
        min_coord_y, max_coord_y = min(float(data[0].x[:, COORDY].min()), float(data[-1].x[:, COORDY].min())) - 0.2, max(float(data[0].x[:, COORDY].max()), float(data[-1].x[:, COORDY].max())) * 1.1 + data[0].u.max()*5
    else:
        min_coord_x, max_coord_x = float(COORDX.min()) - 0.1, float(COORDX.max()) + 0.1
        min_coord_y, max_coord_y = float(COORDY.min()) - 0.1, float(COORDY.max()) + 0.1

    for i in tqdm(range(len(data))):
        sample = data[i]
        variable = sample.x[:, VAR].tolist()[:]
        if 'COORD.COOR1' in state_variables:
            coord_x = sample.x[:, COORDX].tolist()[:]
            coord_y = sample.x[:, COORDY].tolist()[:]
        else:
            coord_x = COORDX.tolist()[:]
            coord_y = COORDY.tolist()[:]

        try:
            if sample.u:
                u = sample.u.squeeze().tolist()[:]
        except:
            u = None
        n = None #sample.n[:]
        steps = i

        if with_edges:
            edge_index = sample.edge_index.tolist()
            # keep_idx = torch.argwhere(edge_index[0] < len(variable)).squeeze()
            # edge_index = edge_index[:, keep_idx]
            # keep_idx = torch.argwhere(edge_index[1, :] < len(variable)).squeeze()
            # edge_index = edge_index[:, keep_idx].tolist()
        else:
            edge_index = None

        # min_coord_x, max_coord_x = -0.2, 1.2
        # min_coord_y, max_coord_y = -0.2, 1.5

        plot_graph_data_plotly(coord_x, coord_y, variable, edge_index=edge_index, n=n, u=u,
                               lim_values=(min_value, max_value), lim_coord_x=(min_coord_x, max_coord_x),
                               lim_coord_y=(min_coord_y, max_coord_y), title=f'{title} {plot_variable} iter={i}',
                               save_path=path / f'iter={i}.png', colorscale=colorscale)

    images = []
    for i in range(steps):
        images.append(Image.open(path / f'iter={i}.png'))
    print(len(images))
    images[0].save(path / 'animation.gif', save_all=True, append_images=images[1:], duration=250, loop=0)

    clip = mp.VideoFileClip(str(path / 'animation.gif'))
    clip.write_videofile(str(path / f'{title}_animation.mp4'))

    for file in path.glob(f"*.png"):
        file.unlink()
        print(f"Deleted: {file}")

    return str(path / f'{title}_animation.mp4')

import matplotlib.pyplot as plt
def plt_matrix(M_big):
    A_cpu = M_big.detach().cpu().numpy()
    vmin = -0.8  # A_cpu.min()
    vmax = 0.8  # A_cpu.max()
    plt.figure(figsize=(45, 45))
    # Visualizamos la matriz como una imagen utilizando Matplotlib
    plt.imshow(A_cpu, cmap='PuOr', vmin=vmin, vmax=vmax)  # Puedes elegir cualquier mapa de colores (cmap)
    plt.colorbar()
    # Guardamos la imagen como un archivo PNG
    plt.savefig('M_big_overfit_elas_.png', dpi=300, bbox_inches='tight')