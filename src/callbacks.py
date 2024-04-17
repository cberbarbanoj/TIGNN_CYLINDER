import os
import torch
import lightning.pytorch as pl
from pathlib import Path

# from src.plots.plots import make_video_predicted_and_ground_truth
import wandb
from copy import deepcopy
import shutil
from src.utils.utils import compute_connectivity
from src.utils.plots import plot_2D, plot_3D, plot_2D_image, plot_image3D, plt_LM, make_gif, plotError
from src.evaluate import roll_out,compute_error, print_error

from lightning.pytorch.callbacks import LearningRateFinder


class HistogramPassesCallback(pl.Callback):

    def on_validation_end(self, trainer, pl_module):
        if (pl_module.current_epoch % 25 == 0 and len(pl_module.error_message_pass) > 0):
            table = wandb.Table(data=pl_module.error_message_pass, columns=["pass", "epoch", "error"])
            trainer.logger.experiment.log(
                {f'error_message_pass': table})


class RolloutCallback(pl.Callback):
    def __init__(self, dataloader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)


    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0 and trainer.current_epoch%pl_module.rollout_freq == 0:
            try:
                z_net, z_gt, q_0 = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.data_dim)
                save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}.gif')
                # save_dir_L = os.path.join(pl_module.save_folder, f'epoch_L_{trainer.current_epoch}.png')
                # save_dir_M = os.path.join(pl_module.save_folder, f'epoch_M_{trainer.current_epoch}.png')
                # plt_LM(L, save_dir_L)
                # plt_LM(M, save_dir_M)
                # trainer.logger.experiment.log({"L": wandb.Image(save_dir_L)})
                # trainer.logger.experiment.log({"M": wandb.Image(save_dir_M)})

                if pl_module.data_dim == 2:
                    plot_2D(z_net, z_gt, save_dir=save_dir, var=0, q_0=q_0)
                else:
                    plot_3D(z_net, z_gt, save_dir=save_dir, var=0)
                trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})
            except:
                print()

    def on_train_end(self, trainer, pl_module):
        z_net, z_gt, q_0 = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.data_dim)
        filePath = os.path.join(pl_module.save_folder, 'metrics.txt')
        save_dir = os.path.join(pl_module.save_folder, f'final_{trainer.current_epoch}.gif')
        with open(filePath, 'w') as f:
            error, L2_list = compute_error(z_net, z_gt,pl_module.state_variables)
            lines = print_error(error)
            f.write('\n'.join(lines))
            print("[Test Evaluation Finished]\n")
            f.close()
        plotError(z_gt, z_net, L2_list, pl_module.state_variables, pl_module.data_dim, pl_module.save_folder)
        if pl_module.data_dim == 2:
            plot_2D(z_net, z_gt, save_dir=save_dir, var=0, q_0=q_0)
            plot_2D_image(z_net, z_gt, -1, var=0, q_0=q_0, output_dir=pl_module.save_folder)
        else:
            plot_3D(z_net, z_gt, save_dir=save_dir, var=0)
            data = [sample for sample in self.dataloader]
            plot_image3D(z_net, z_gt, pl_module.save_folder, var=0, step=-1, n=data[0].n)
        shutil.copyfile(os.path.join('src', 'gnn_nodal.py'), os.path.join(pl_module.save_folder, 'gnn_nodal.py'))
        shutil.copyfile(os.path.join('data', 'jsonFiles', 'dataset_cfd.json'),
                        os.path.join(pl_module.save_folder, 'dataset_cfd.json'))



class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class MessagePassing(pl.Callback):
    def __init__(self, dataloader, rollout_variable=None, rollout_freq=None,
                 rollout_simulation=None, **kwargs):
        super().__init__(**kwargs)

        self.rollout_variable = rollout_variable
        self.rollout_freq = 25
        if rollout_simulation is None:
            self.rollout_simulation = [0]
            self.rollout_gt = {0: []}
        else:
            self.rollout_simulation = rollout_simulation
            self.rollout_gt = {sim: [] for sim in range(len(rollout_simulation))}

        for sample in dataloader:
            if rollout_simulation is None:
                self.rollout_gt[0].append(sample)
            else:
                self.rollout_gt[int(sample.idx)].append(sample)

    def __clean_artifacts(self, trainer):
        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.__clean_artifacts(trainer)

    def on_validation_epoch_end(self, trainer, pl_module, index=0):

        if ((pl_module.current_epoch + 1) % self.rollout_freq == 0) and (pl_module.current_epoch > 0):

            for sim, rollout_gt in self.rollout_gt.items():
                iter_step = len(rollout_gt)//2
                print(f'\nMessage passing Iter={iter_step} Sim={self.rollout_simulation[sim]}')
                if torch.cuda.is_available():
                    z_pred, z_t1, z_message_pass = pl_module.predict_step(rollout_gt[index].to('cuda'), 1, passes_flag=True)
                else:
                    z_pred, z_t1, z_message_pass = pl_module.predict_step(rollout_gt[index].to('cpu'), 1, passes_flag=True)
                pl_module.position_index = 1
                # Make video out of data pred and gt
                message_pass = []
                z_first_coords = z_message_pass[0][1]
                for i in range(len(z_message_pass)):
                    sample = deepcopy(rollout_gt[index])
                    # current coords
                    z_current_coords = z_message_pass[i][1]
                    # compute coords norm error
                    z_current_coords_error = torch.norm(z_current_coords-z_first_coords, dim=1).unsqueeze(-1)
                    # z_current_coords_error = (z_current_coords[:,1]-z_first_coords[:,1]).unsqueeze(-1)
                    # add norm error as state variable for plot
                    sample.x = torch.cat([z_message_pass[i][1], z_current_coords_error], dim=1)
                    # specific u imposed
                    sample.u = z_message_pass[i][2]
                    message_pass.append(sample.to('cpu'))

                state_variables = deepcopy(pl_module.state_variables)
                state_variables.append('message pass flow')

                path_to_video_message_pass_coord2 = make_gif(message_pass,
                                                       Path(
                                                           trainer.checkpoint_callback.dirpath) / 'videos',
                                                       f'',
                                                       plot_variable='message pass flow',
                                                       state_variables=state_variables,
                                                       with_edges=True,
                                                             colorscale='YlOrRd')

                trainer.logger.experiment.log({f"Passes Message {self.rollout_simulation[sim]}": wandb.Video(path_to_video_message_pass_coord2, format='mp4')})