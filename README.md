# 2025 人工智能实践大作业

## 环境配置

```bash
pip install -r requirements.txt

# CIFAT-10 数据集
mkdir data/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/cifar-10-python.tar.gz
tar -xf data/cifar-10-python.tar.gz -C data/
```

预期得到文件结构

```
data/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── readme.html
    └── test_batch
```

## 运行

### CIFAR-10

**使用 Attention 架构**

```bash
python weno/train_CIFAR_BagDistillation_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py --epochs <epochs_num> --seed <random_seed>
```

**使用 DSMIL 架构**

```bash
python weno/train_CIFAR_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py --epochs <epochs_num> --seed <random_seed>
```

## 可视化

```bash
tensorboard --logdir=runs/
```