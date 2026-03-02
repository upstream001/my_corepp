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
8.评价
python evaluate_strawberry_results.py \
    --pred_dir /home/tianqi/my_corepp/logs/strawberry/test_results \
    --gt_dir /home/tianqi/my_corepp/data/20260301_dataset/complete \
    --split /home/tianqi/my_corepp/deepsdf/experiments/splits/strawberry_test.json \
    --threshold 0.01


Attn:如果使用论文的数据增强脚本，流程如下：
1.获取deepsdf训练数据
python data_preparation/prepare_deepsdf_training_data.py --src ./data/potato_augmented
2.针对augmented数据集python make_splits.py
3.训练deepsdf
python train_deep_sdf.py --experiment deepsdf/experiments/strawberry
4.训练encoder
python make_splits.py针对原来的数据集
python train_strawberry_pcd.py --config configs/strawberry.json
