# Hands Dirty on DL

## 介绍
学习Deep Learning，光读论文和看书终觉浅，这里把我学习Deep Learning所写的一些代码和Notebook记录下来，方便查阅。

## Setup

### 步骤一. 环境配置

#### 使用`conda`
```bash
conda env create -f conda_env.yml
conda activate pytorch-cu124
```

**Q1. 如果在VSCode terminal的prompt看到多个conda env，比如`(pytorch-cu124) (base)`，实际调用的python路径没有指向环境`pytorch-cu124`的python**

    使用如下命令关闭自动激活`base`
    ```base
    conda config --set auto_activate_base False
    ```

#### 使用`pip`
```bash
pip install -r requirements.txt
```

#### 输出环境配置

```bash
# 使用pip
pip freeze > requirements.txt
# 使用conda
conda export > conda_env.yml
```

### 步骤二. 执行可编辑安装
这一步将package`hdd`至于python路径之下。
```bash
# 在hands-dirty-on-dl目录下运行该命令
pip install -e .
# 此时，你可以运行pytest进行测试
pytest
```

## Repo结构

* `dataset`: 用于存放模型用到的数据集
* `hdd`: 存放python代码
  * `data_util`: 数据相关的Util函数,比如某些数据transforms
  * `dataset`: 数据集代码
  * `device`: device相关代码
  * `models`: 模型代码
  * `train`: 训练用到的util函数
  * `visualization`： 可视化相关代码
* `notebooks`： 存放notebook
  * `classification` 分类任务相关的notebook

