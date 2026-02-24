import os
import open3d as o3d
import numpy as np
import json
import random
from tqdm import tqdm
from dataloaders.cameralaser_w_masks import generate_deepsdf_target

def process(input_dir, files, split_dict, out_dir_base):
    for f in tqdm(files):
        name = f[:-4]
        split_dict.append(name)
        # DeepSDF expects class_name/instance_name/laser/samples.npz
        out_dir = os.path.join(out_dir_base, "Strawberry", name, "laser")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "samples.npz")
        
        if not os.path.exists(out_file):
            pcd = o3d.io.read_point_cloud(os.path.join(input_dir, f))
            # Generate positive and negative SDF targets along estimated normals and free space
            pos_tensor, neg_tensor = generate_deepsdf_target(pcd, align_with=np.array([0.0, 1.0, 0.0]))
            np.savez(out_file, pos=pos_tensor, neg=neg_tensor)

def main():
    data_source = "/home/tianqi/my_corepp/data"
    input_dir = os.path.join(data_source, "strawberry", "complete")
    out_dir_base = os.path.join(data_source, "SdfSamples")
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.ply')])
    random.seed(42)
    random.shuffle(files)
    
    train_split = int(len(files) * 0.8)
    train_files = files[:train_split]
    test_files = files[train_split:]
    
    train_list = []
    test_list = []
    
    print("Processing Train Set...")
    process(input_dir, train_files, train_list, out_dir_base)
    print("Processing Test Set...")
    process(input_dir, test_files, test_list, out_dir_base)
    
    train_json = {"StrawberryDataset": {"Strawberry": train_list}}
    test_json = {"StrawberryDataset": {"Strawberry": test_list}}
    
    splits_dir = "/home/tianqi/my_corepp/deepsdf/experiments/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, "strawberry_train.json"), "w") as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(splits_dir, "strawberry_test.json"), "w") as f:
        json.dump(test_json, f, indent=2)
        
    print("SDF data preparation completed.")

if __name__ == '__main__':
    main()
