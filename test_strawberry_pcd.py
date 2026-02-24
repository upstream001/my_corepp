#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import time

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
from networks.models import PointCloudEncoder
from dataloaders.strawberry_pcd import StrawberryPcdDataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--save_dir", type=str, default="logs/strawberry/test_results", help="Dir to save reconstructed meshes")
    parser.add_argument("--resolution", type=int, default=128, help="Marching cubes resolution")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        param = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load DeepSDF Decoder
    print("Loading DeepSDF Decoder...")
    experiment_directory = "/home/tianqi/my_corepp/deepsdf/experiments/strawberry"
    with open(os.path.join(experiment_directory, 'specs.json'), 'r') as f:
        specs = json.load(f)

    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    
    # Check if we should use latest or specific epoch for decoder
    decoder_model_path = os.path.join(experiment_directory, ws.model_params_subdir, str(specs["NumEpochs"]) + ".pth")
    if not os.path.exists(decoder_model_path):
        decoder_model_path = os.path.join(experiment_directory, ws.model_params_subdir, "latest.pth")

    saved_model_state = torch.load(decoder_model_path)
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder.to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # 2. Load Network (Point Cloud Encoder)
    print("Loading Point Cloud Encoder...")
    if param['encoder'] == 'point_cloud':
        encoder = PointCloudEncoder(3, latent_size).to(device)
    else:
        raise ValueError("Only point_cloud encoder supported in this script.")
        
    encoder_weight_path = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    if not os.path.exists(encoder_weight_path):
        raise FileNotFoundError(f"Encoder weights not found at {encoder_weight_path}. Did you finish training?")
        
    encoder.load_state_dict(torch.load(encoder_weight_path, map_location=device))
    encoder.eval()

    # 3. Load Test Dataset
    dataset_test = StrawberryPcdDataset(param['data_dir'], split='test', num_points=param['input_size'], supervised_3d=False)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    print(f"Starting testing on {len(dataset_test)} samples...")
    start_all = time.time()
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(loader_test)):
            fruit_id = item['fruit_id'][0] # batch_size=1
            encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)
            
            # encode partial point cloud to predict latent code
            pred_latent = encoder(encoder_input)
            
            # Use deep_sdf.mesh module to run marching cubes
            mesh_filename = os.path.join(args.save_dir, fruit_id) # no .ply ext, deepsdf adds it automatically
            
            # Reconstruct Mesh
            deep_sdf.mesh.create_mesh(
                decoder, 
                pred_latent, 
                mesh_filename, 
                start=time.time(), 
                N=args.resolution, 
                max_batch=int(2 ** 18)
            )

    print(f"Testing finished! Total time: {time.time() - start_all:.1f}s")
    print(f"Reconstructed models saved in: {args.save_dir}")

if __name__ == '__main__':
    main()
