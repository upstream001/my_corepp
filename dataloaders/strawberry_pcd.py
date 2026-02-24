import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import logging

class StrawberryPcdDataset(Dataset):
    def __init__(self, data_source, split='train', overfit=False, num_points=2048, supervised_3d=True, latents_dir=None):
        self.data_source = data_source
        self.overfit = overfit
        self.num_points = num_points
        self.supervised_3d = supervised_3d
        
        split_path = os.path.join("/home/tianqi/my_corepp/deepsdf/experiments/splits", f"strawberry_{split}.json")
        with open(split_path, 'r') as f:
            data = json.load(f)
        self.split_ids = data['StrawberryDataset']["Strawberry"]
        
        if self.overfit:
            self.split_ids = self.split_ids[:2]
            
        # Extract pre-trained Latent keys 
        # train_deep_sdf.py saves the optimized latents into epoch files: latents = torch.load('..')['latent_codes']
        self.latents = None
        if self.supervised_3d and latents_dir and os.path.exists(latents_dir):
            if latents_dir.endswith('.pth'): 
                # This handles passing a specific epoch checkpoint from LatentCodes
                latents_file = torch.load(latents_dir)
                logging.info(f"Loaded latent vectors from {latents_dir}")
                self.latents = latents_file['latent_codes']['weight'] 
            else:
                 # Default logic for individual pth files if applicable
                 pass

    def sample_pcd(self, pcd, num):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return np.zeros((num, 3), dtype=np.float32)
        if len(points) >= num:
            idx = np.random.choice(len(points), num, replace=False)
        else:
            idx = np.random.choice(len(points), num, replace=True)
        return points[idx]

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        instance_name = self.split_ids[idx]
        
        # registered partial point clouds and complete point clouds directly mapped
        partial_pcd_path = os.path.join(self.data_source, "strawberry/partial", f"{instance_name}.ply")
        target_pcd_path = os.path.join(self.data_source, "strawberry/complete", f"{instance_name}.ply")
        
        partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
        target_pcd = o3d.io.read_point_cloud(target_pcd_path)
        
        partial_points = self.sample_pcd(partial_pcd, self.num_points)
        target_points = self.sample_pcd(target_pcd, self.num_points)
        
        item = {
             'fruit_id': instance_name,
             'partial_pcd': torch.from_numpy(partial_points).float(),
             'target_pcd': torch.from_numpy(target_points).float()
        }
        
        if self.supervised_3d and self.latents is not None:
             # DeepSDF maintains latents in the exact sequence they exist in the splits: class_name/instance_name.
             # So the index `idx` maps 1-1 with the latent rows in the matrix
             item['latent'] = self.latents[idx].squeeze()
            
        return item
