#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import pathlib
import open3d as o3d
import numpy as np
import copy
import argparse
import json
from tqdm import tqdm

# 加载 SDF 采样函数
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from prepare_deepsdf_training_data import generate_tsdf_samples

def main():
    parser = argparse.ArgumentParser(description="Strawberry Data Augmentation (Compatible with one-folder structure)")
    parser.add_argument("--config", default="data_preparation/augment.json", help="json filename with the parameters")
    parser.add_argument("--src", required=True, help="Path to your 'complete' folder containing .ply files")
    parser.add_argument("--dst", required=True, help="Path to store the augmented dataset (will follow DeepSDF structure)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)

    # 获取所有 ply 文件
    ply_files = [f for f in os.listdir(args.src) if f.endswith('.ply')]
    print(f"Found {len(ply_files)} items in {args.src}")

    for ply_name in tqdm(ply_files):
        item_id = os.path.splitext(ply_name)[0]
        pcd_path = os.path.join(args.src, ply_name)
        pcd = o3d.io.read_point_cloud(pcd_path)

        for jdx in range(config['no_of_augmentations']):
            # 1. 随机缩放 (Scale)
            scale = np.random.uniform(config['min_scalefactor'], config['max_scalefactor'], size=(3,))
            T_scale = np.diag([scale[0], scale[1], scale[2], 1.0])

            # 2. 绕 Z 轴随机旋转 (Rotation Z Only)
            angle = np.random.uniform(-config['max_rotation_angle_degree'] * np.pi/180.0,
                                      +config['max_rotation_angle_degree'] * np.pi/180.0)
            R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, angle]))
            T_R = np.eye(4)
            T_R[0:3, 0:3] = R

            # 3. 随机剪切 (Shear)
            shear = np.random.uniform(-config['max_shear'], +config['max_shear'], size=(2,))
            T_shear = np.eye(4)
            T_shear[0,1] = shear[0]
            T_shear[0,2] = shear[1]

            # 合并变换矩阵
            T = T_shear @ T_R @ T_scale

            # 应用变换
            tmp = copy.deepcopy(pcd)
            tmp.transform(T)

            # 重新估计法线并保证向外 (SDF 采样非常重要)
            tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            tmp.orient_normals_towards_camera_location() 
            tmp.normals = o3d.utility.Vector3dVector(-np.array(tmp.normals))

            # 生成 SDF 采样点 (用于 DeepSDF 训练)
            # 使用项目推荐的 tsdf 阈值 (0.02, 0.01)
            swl_points = np.hstack((np.array(tmp.points), np.array(tmp.normals)))
            no_of_samples = 100000
            no_samples_per_point = int(np.ceil(no_of_samples / swl_points.shape[0]))
            
            (pos, neg) = generate_tsdf_samples(
                swl_points, 
                no_samples_per_point=no_samples_per_point,
                tsdf_positive=0.02,
                tsdf_negative=0.01
            )
            
            # 随机选取固定数量
            pos = pos[np.random.choice(pos.shape[0], no_of_samples, replace=False), :]
            neg = neg[np.random.choice(neg.shape[0], no_of_samples, replace=False), :]

            # 保存结果（DeepSDF 格式：物体名_编号/laser/xxx）
            output_dir = os.path.join(args.dst, f"{item_id}_aug_{jdx:03d}", "laser")
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            o3d.io.write_point_cloud(os.path.join(output_dir, 'fruit.ply'), tmp)
            np.savez(os.path.join(output_dir, 'samples.npz'), pos=pos, neg=neg)

if __name__ == "__main__":
    main()
