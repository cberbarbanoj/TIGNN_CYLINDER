import os
import json
import argparse
import datetime
import torch

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataLoader.dataset import GraphDataset
from src.dataLoader.datamodule import DatasetDPRollout, DatasetDP
from src.gnn_dd import PlasticityGNN
from src.callbacks import RolloutCallback, FineTuneLearningRateFinder, HistogramPassesCallback, MessagePassing
from src.utils.utils import str2bool
from src.evaluate import generate_results


if __name__ == '__main__':

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--transfer_learning', default=False, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'epoch=21-val_loss=0.01.ckpt', type=str, help='name')

    # Dataset Parametersa
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    # parser.add_argument('--dset_name', default='d6_waterk10_noTensiones_radius_.pt', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_cylinder.json', type=str, help='dataset directory')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_dir_exp', default=r'outputs/', type=str,
                        help='output directory')
    parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')
    parser.add_argument('--experiment_name', default='exp3', type=str, help='experiment output name tensorboard')
    args = parser.parse_args()  # Parse command-line arguments

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    f = open(os.path.join(args.dset_dir, 'jsonFiles', args.dset_name))
    dInfo = json.load(f)

    pl.seed_everything(dInfo['model']['seed'], workers=True)
    # train_set = GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['train']), short=False)
    train_set = DatasetDP(dInfo, './data', split='train', trajectory=[80, 81, 82, 83, 84])
    # train_set = DatasetDP(dInfo, './data', split='train', trajectory=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    # train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])
    train_dataloader = train_set.get_loader(batch_size=dInfo['model']['batch_size'], shuffle=True)

    scaler_var, scaler_pos = train_set.get_stats()

    # Logger
    # val_set = GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['val']), short=False)
    val_set = DatasetDP(dInfo,'./data',  split='train', trajectory=[79, 85])
    # val_set = DatasetDP(dInfo,'./data',  split='valid', trajectory=[1, 2, 3, 4, 5])
    # val_dataloader = DataLoader(val_set, batch_size=dInfo['model']['batch_size'])
    val_dataloader = val_set.get_loader(batch_size=dInfo['model']['batch_size'], shuffle=True)
    # test_set = GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['test']))
    test_set = DatasetDPRollout(dInfo,'./data',  split='train', trajectory=82)
    # test_dataloader = DataLoader(test_set, batch_size=1)
    test_dataloader = test_set.get_loader(batch_size=1)

    # name = f"yesfN_pretrain_edgeAu02_decoYeaddmean_yj_v6_rc9_hiddenDim{dInfo['model']['dim_hidden']}_NumLayers{dInfo['model']['n_hidden']}_Filters{dInfo['model']['filters']}_Passes{dInfo['model']['passes']}_lr{dInfo['model']['lr']}_noise{dInfo['model']['noise_var']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    name = f"pretrain_batch1_NumLayers{dInfo['model']['n_hidden']}_Passes{dInfo['model']['passes']}_lr{dInfo['model']['lr']}_noise{dInfo['model']['noise_var']}_lamda{dInfo['model']['lambda_d']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # name = f"prueba_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_folder = f'outputs/runs/{name}'
    wandb_logger = WandbLogger(name=name, project=dInfo['project_name'])
    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(dirpath=save_folder,  filename='{epoch}-{val_loss:.2f}', monitor='val_loss', save_top_k=3)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rollout = RolloutCallback(test_dataloader)
    message_passing = MessagePassing(test_dataloader)
    passes_tracker = HistogramPassesCallback()

    # Instantiate model
    plasticity_gnn = PlasticityGNN(train_set.dims, scaler_var, scaler_pos, dInfo, save_folder)
    print(plasticity_gnn)
    wandb_logger.watch(plasticity_gnn)

    # load weights
    if args.transfer_learning:
        path_checkpoint = os.path.join(args.dset_dir, 'weights', args.pretrain_weights)
        checkpoint_ = torch.load(path_checkpoint, map_location=device)
        plasticity_gnn.load_state_dict(checkpoint_['state_dict'], strict=False)

    # Set Trainer
    trainer = pl.Trainer(accelerator="gpu",
                         # accumulate_grad_batches=7,
                         logger=wandb_logger,
                         # callbacks=[checkpoint, lr_monitor, FineTuneLearningRateFinder(milestones=(5, 10)), rollout, passes_tracker, early_stop],
                         callbacks=[checkpoint, lr_monitor, rollout, message_passing, early_stop, passes_tracker],
                         profiler="simple",
                         gradient_clip_val=0.5,
                         num_sanity_val_steps=0,
                         max_epochs=dInfo['model']['max_epoch'],
                         # precision="64-true",
                         deterministic=True, #Might make your system slower,
                         fast_dev_run=False)#, # debugging purposes

    # Train model
    trainer.fit(model=plasticity_gnn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # test_set = GraphDataset(dInfo,
    #                         os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['test']))
    # test_dataloader = DataLoader(test_set, batch_size=dInfo['model']['batch_size'])
    #
    # generate_results(trainer, plasticity_gnn, test_dataloader, os.path.join(args.output_dir_exp, args.experiment_name))
