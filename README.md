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
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2.Datasets
以下提供官网下载和百度网盘下载

1）ModelNet40数据集下载：

官网下载：[ModelNet40](https://modelnet.cs.princeton.edu/#)

百度网盘：[链接](https://pan.baidu.com/s/1TF7vgUGOih5aOL3Tjhrc9A )
提取码：a0ev

2）ScanObjectNN数据集下载：

官网下载：[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)

百度网盘：[链接](https://pan.baidu.com/s/1WNtUL7s4m_bi6zHl7c7exg)
提取码：y62f

3）ShapeNetCore v2数据集下载

官网下载：[ShapeNetCore v2](https://shapenet.org/download/shapenetcore)

本文推荐使用点云数据集仓库[PointCloudDatasets](https://github.com/antao97/PointCloudDatasets),该仓库包含百度网盘链接。

以下示例默认使用1024点，和坐标信息，不包含法向量，可以手动改变需要的参数进行训练和测试
## 3.Train
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --log_dor <log_dir> --learning_rate 0.02 --dim 768 --heads 12 --dataset <specify_dataset>
```

## 3.Test
```
CUDA_VISIBLE_DEVICES=<gpu_device> python test_classification.py --model pct --log_dor <log_dir>  --dim 768 --heads 12 --dataset <specify_dataset>
```
## 4.Pre_training
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --log_dor <log_dir> --learning_rate 0.02 --dim 768 --heads 12 --dataset shapenetcorev2
```

## 5.Fine_tuning
```
CUDA_VISIBLE_DEVICES=<gpu_device> python train_classification.py --model pct --log_dor <log_dir> --learning_rate 0.02 --dim 768 --heads 12 --dataset <specify_dataset>
```
