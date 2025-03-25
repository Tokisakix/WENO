# 2025 人工智能实践大作业

## 环境配置

```bash
pip install -r requirements.txt
```

### CIFAR-10 数据集

```bash
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

### TCGA-KIRC 数据集

从[此处](https://drive.google.com/file/d/1YJQewp9sLjXcrxSEdVQkOmHHt6VxIIAq/view?usp=drive_link)下载数据压缩包，解压存放至 `data/tcga-kirc`

预期得到文件结构

```
data/
└── tcga-kirc/
    ├── batch_blood.npz
    ├── batch_cancer.npz
    ├── batch_empty.npz
    ├── batch_normal.npz
    ├── batch_other.npz
    └── batch_stroma.npz
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

### TCGA-KIRC

**使用 Attention 架构**

```bash
python weno/train_TCGA_KIRC_BagDistillation_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py --epochs <epochs_num> --seed <random_seed>
```

**使用 DSMIL 架构**

```bash
python weno/train_TCGA_KIRC_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py --epochs <epochs_num> --seed <random_seed>
```

## 可视化

```bash
tensorboard --logdir=runs/
```