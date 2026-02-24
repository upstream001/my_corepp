#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
from networks.models import Encoder, PointCloudEncoder
from dataloaders.strawberry_pcd import StrawberryPcdDataset
# from loss import Loss, RepellingLoss, AttRepLoss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        param = json.load(f)

    os.makedirs(param["checkpoint_dir"], exist_ok=True)
    os.makedirs(param["log_dir"], exist_ok=True)
    writer = SummaryWriter(param["log_dir"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DeepSDF Decoder
    experiment_directory = "/home/tianqi/my_corepp/deepsdf/experiments/strawberry"
    with open(os.path.join(experiment_directory, 'specs.json'), 'r') as f:
        specs = json.load(f)

    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(os.path.join(experiment_directory, ws.model_params_subdir, specs["NumEpochs"].__str__() + ".pth"))
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder.to(device)
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()

    # Load Network (Point Cloud Encoder)
    if param['encoder'] == 'point_cloud':
        encoder = PointCloudEncoder(3, latent_size).to(device)
    else:
        raise ValueError("Only point_cloud encoder supported in this pipeline.")

    # Datasets
    latents_pth = os.path.join(experiment_directory, ws.latent_codes_subdir, specs["NumEpochs"].__str__() + ".pth")
    dataset_train = StrawberryPcdDataset(param['data_dir'], split='train', num_points=param['input_size'], supervised_3d=param['supervised_3d'], latents_dir=latents_pth)
    loader_train = DataLoader(dataset_train, batch_size=param['batch_size'], shuffle=True, drop_last=True)
    
    dataset_test = StrawberryPcdDataset(param['data_dir'], split='test', num_points=param['input_size'], supervised_3d=False) # Wait: evaluation usually compares outputs directly. Or uses 3d loss.
    loader_test = DataLoader(dataset_test, batch_size=param['batch_size'], shuffle=False)

    params = list(encoder.parameters())
    optim = torch.optim.Adam(params, lr=param["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.97)

    n_iter = 0
    criterion = torch.nn.MSELoss()

    for e in range(param["epoch"]):
        encoder.train()
        for idx, item in enumerate(tqdm(loader_train, desc=f"Epoch {e+1}/{param['epoch']}")):
            n_iter += 1
            optim.zero_grad()
            
            # Since point clouds are already registered and ready
            encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) 
            
            latent_batch = encoder(encoder_input)
            
            loss = 0
            if param["supervised_3d"]:
                loss_3d = criterion(latent_batch, item['latent'].to(device))
                loss += loss_3d
                writer.add_scalar('Loss/Train/L2_latent', loss_3d.item(), n_iter)

            # additional losses like repulsion or SDF could be added here
            # using decoder...

            loss.backward()
            optim.step()

        scheduler.step()
        
        # Save checkpoints
        if (e + 1) % param['checkpoint_frequency'] == 0:
            torch.save(encoder.state_dict(), os.path.join(param['checkpoint_dir'], f"encoder_ep{e+1}.pt"))

    # save final
    torch.save(encoder.state_dict(), os.path.join(param['checkpoint_dir'], param['checkpoint_file']))
    print("Training finished!")

if __name__ == '__main__':
    main()
