1.草莓数据集
2.python prepare_deepsdf_training_data.py --src ../data/strawberry/complete
来准备deepsdf的训练数据
3.训练deepsdf
python make_splits.py
python train_deep_sdf.py --experiment deepsdf/experiments/strawberry
4.训练encoder
python train_strawberry_pcd.py --config configs/strawberry.json
5.测试
python test_strawberry_pcd.py --config configs/strawberry.json
6.可视化
python visualize_dir.py /home/tianqi/my_corepp/logs/strawberry/test_results
python visualize_mesh.py /home/tianqi/my_corepp/logs/strawberry/test_results
7.tensorboard查看
tensorboard --logdir /home/tianqi/my_corepp/logs/strawberry/runs
