# Transformer建模多尺度几何关系的点云识别模型

## 1.Requirements
我们在以下环境中运行并测试本代码
Linux系统;
Pytorch==1.8;
Python==3.7;
CUDA==10.1;
torchvision;

```
pip install -r requirements.txt
```

还需要安装必要的工具库
```
pip install pointnet2_ops_lib  # 该库提供furthest_point_sample和gather_option函数
```

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl # 该库提供KNN函数
```

## 2.Datasets
以下提供官网下载和百度网盘下载

1）ModelNet40数据集下载：

官网下载：[ModelNet40](https://modelnet.cs.princeton.edu/#)


2）ScanObjectNN数据集下载：

官网下载：[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)


3）ShapeNetCore v2数据集下载

官网下载：[ShapeNetCore v2](https://shapenet.org/download/shapenetcore)

本文推荐使用点云数据集仓库[PointCloudDatasets](https://github.com/antao97/PointCloudDatasets),该仓库包含百度网盘链接。

以下示例默认使用1024点，和坐标信息，不包含法向量，可以手动改变需要的参数进行训练和测试
## 3.Train
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --log_dor <log_dir> --learning_rate 0.02 --dim 768 --heads 12 --dataset <specify_dataset>
```

## 4.Test
```
CUDA_VISIBLE_DEVICES=<gpu_device> python test_classification.py --model pct --log_dor <log_dir>  --dim 768 --heads 12 --dataset <specify_dataset>
```
## 5.Pre_training
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --log_dor <log_dir>  --dim 768 --heads 12 --dataset shapenetcorev2
```

## 6.Fine_tuning
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --use_pretrin --log_dor <log_dir> --learning_rate 0.02 --dim 768 --heads 12 --dataset <specify_dataset>
```
## 7.Robustness test
指定需要何种鲁棒性测试，例如添加高斯噪声需要在下面命令中添加--guass_noise参数，默认sigma=0.01。
```
UDA_VISIBLE_DEVICES=<gpu_device> python test_classification.py --model pct --guass_noise --log_dor <log_dir>  --dim 768 --heads 12 --dataset <specify_dataset>
```
在作丢失点的实验时，需要添加使用点的个数例如  --num_points 768
```
UDA_VISIBLE_DEVICES=<gpu_device> python test_classification.py --model pct --num_points768 --log_dor <log_dir>  --dim 768 --heads 12 --dataset <specify_dataset>
```

## 最后我们还提供在较大数据集ShapeNetCore v2数据集上预训练模型。[ShapeNetCore_v2_pre_train](https://pan.baidu.com/s/1IiApcvwfW5oeFydjFIMjxA )提取码：rvga
注意：在进行微调时，不要添加--use_normals参数。
