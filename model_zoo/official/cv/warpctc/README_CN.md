# 目录

<!-- TOC -->

- [目录](#目录)
    - [WarpCTC描述](#warpctc描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
            - [训练脚本参数](#训练脚本参数)
        - [参数配置](#参数配置)
    - [数据集准备](#数据集准备)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
        - [推理性能](#推理性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## WarpCTC描述

以下为MindSpore中用自生成的验证码图像数据集来训练WarpCTC的例子。

## 模型架构

WarpCTC是带有一层FC神经网络的二层堆叠LSTM模型。详细信息请参见src/warpctc.py。

## 数据集

该数据集由第三方库[captcha](https://github.com/lepture/captcha)自行生成，可以在图像中随机生成数字0至9。在本网络中，我们设置数字个数为1至4。

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend,GPU或者CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

- 生成数据集

    执行脚本`scripts/run_process_data.sh`生成数据集。默认情况下，shell脚本将分别生成10000个测试图片和50000个训练图片。

    ```text
     $ cd scripts
     $ sh run_process_data.sh

     # 执行后，数据集如下：
     .  
     └─warpctc
       └─data
         ├─ train  # 训练数据集
         └─ test   # 评估数据集
    ```

- 数据集准备完成后，您可以开始执行训练或评估脚本，具体步骤如下：

    - Ascend处理器环境运行

    ```bash
    # Ascend分布式训练示例
    $ bash run_distribute_train.sh rank_table.json ../data/train

    # Ascend评估示例
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt Ascend

    # Ascend中单机训练示例
    $ bash run_standalone_train.sh ../data/train Ascend
    ```

    在分布式训练中，JSON格式的HCCL配置文件需要提前创建。

    详情参见如下链接：

    [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

    - 在GPU环境运行

    ```bash
    # GPU分布式训练示例
    $ bash run_distribute_train_for_gpu.sh 8 ../data/train

    # GPU单机训练示例
    $ bash run_standalone_train.sh ../data/train GPU

    # GPU评估示例
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt GPU
    ```

    - 在CPU环境运行

    ```bash
    # CPU训练示例
    $ bash run_standalone_train.sh ../data/train CPU
    或者
    python train.py --train_data_dir=./data/train --device_target=CPU

    # CPU评估示例
    $ bash run_eval.sh ../data/test warpctc-30-97.ckpt CPU
    或者
    python eval.py --test_data_dir=./data/test --checkpoint_path=warpctc-30-97.ckpt --device_target=CPU
    ```

    - 在ModelArts上运行
      如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/)
        - 在ModelArt上使用8卡训练

          ```python
          # (1) 上传你的代码到 s3 桶上
          # (2) 在ModelArts上创建训练任务
          # (3) 选择代码目录 /{path}/warpctc
          # (4) 选择启动文件 /{path}/warpctc/train.py
          # (5) 执行a或b
          #     a. 在 /{path}/warpctc/default_config.yaml 文件中设置参数
          #         1. 设置 ”run_distributed=True“
          #         2. 设置 ”enable_modelarts=True“
          #         3. 如果数据采用zip格式压缩包的形式上传，设置 ”modelarts_dataset_unzip_name={filenmae}"
          #     b. 在 网页上设置
          #         1. 添加 ”run_distributed=True“
          #         2. 添加 ”enable_modelarts=True“
          #         3. 如果数据采用zip格式压缩包的形式上传，添加 ”modelarts_dataset_unzip_name={filenmae}"
          # (6) 上传你的 数据/数据zip压缩包 到 s3 桶上
          # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径（该路径下仅有 数据/数据zip压缩包）
          # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
          # (9) 在网页上的’资源池选择‘项目下， 选择8卡规格的资源
          # (10) 创建训练作业
          ```

        - 在ModelArts上使用单卡验证

          ```python
          # (1) 上传你的代码到 s3 桶上
          # (2) 在ModelArts上创建训练任务
          # (3) 选择代码目录 /{path}/warpctc
          # (4) 选择启动文件 /{path}/warpctc/eval.py
          # (5) 执行a或b
          #     a. 在 /path/warpctc 下的default_config.yaml 文件中设置参数
          #         1. 设置 ”enable_modelarts=True“
          #         2. 设置 “checkpoint_path={checkpoint_path}”({checkpoint_path}表示待评估的 权重文件 相对于 eval.py 的路径,权重文件须包含在代码目录下。)
          #         3. 如果数据采用zip格式压缩包的形式上传，设置 ”modelarts_dataset_unzip_name={filenmae}"
          #     b. 在 网页上设置
          #         1. 设置 ”enable_modelarts=True“
          #         2. 设置 “checkpoint_path={checkpoint_path}”({checkpoint_path}表示待评估的 权重文件 相对于 eval.py 的路径,权重文件须包含在代码目录下。)
          #         3. 如果数据采用zip格式压缩包的形式上传，设置 ”modelarts_dataset_unzip_name={filenmae}"
          # (6) 上传你的 数据/数据zip压缩包 到 s3 桶上
          # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径（该路径下仅有 数据/数据zip压缩包）
          # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
          # (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
          # (10) 创建训练作业
          ```

## 脚本说明

### 脚本及样例代码

```text
.
└──warpctc
  ├── README.md                         # warpctc文档说明
  ├── README_CN.md                      # warpctc中文文档说明
  ├── ascend310_infer                   # 用于310推理
  ├── script
    ├── run_distribute_train.sh         # 启动Ascend分布式训练（8卡）
    ├── run_distribute_train_for_gpu.sh # 启动GPU分布式训练
    ├── run_eval.sh                     # 启动评估
    ├── run_infer_310.sh                # 启动310推理
    ├── run_process_data.sh             # 启动数据集生成
    └── run_standalone_train.sh         # 启动单机训练（1卡）
  ├── src
    ├── model_utils
      ├── config.py                     # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py             # 区分本地/ModelArts训练
      ├── local_adapter.py              # 本地训练获取相关环境变量
      └── moxing_adapter.py             # ModelArts训练获取相关环境变量、交换数据
    ├── dataset.py                      # 数据预处理
    ├── loss.py                         # CTC损失定义
    ├── lr_generator.py                 # 生成每个步骤的学习率
    ├── metric.py                       # warpctc网络准确指标
    ├── warpctc.py                      # warpctc网络定义
    └── warpctc_for_train.py            # 带梯度、损失和梯度剪裁的warpctc网络
  ├── default_config.yaml               # 参数配置
  ├── export.py                         # 推理
  ├── mindspore_hub_conf.py             # Mindspore Hub接口
  ├── eval.py                           # 评估网络
  ├── process_data.py                   # 数据集生成脚本
  ├── postprocess.py                    # 310推理后处理脚本
  ├── preprocess.py                     # 将数据前处理脚本
  └── train.py                          # 训练网络
```

### 脚本参数

#### 训练脚本参数

```bash
# Ascend分布式训练
用法: bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]

# GPU分布式训练
用法： bash run_distribute_train_for_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]

# 单机训练
用法： bash run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_TARGET]
```

### 参数配置

在default_config.yaml中可以同时配置训练参数和评估参数。

```text
max_captcha_digits: 4                       # 每张图像的数字个数上限。
captcha_width: 160                          # captcha图片宽度。
captcha_height: 64                          # capthca图片高度。
batch_size: 64                              # 输入张量批次大小。
epoch_size: 30                              # 只对训练有效，推理固定值为1。
hidden_size: 512                            # LSTM层隐藏大小。
learning_rate: 0.01                         # 初始学习率。
momentum: 0.9                               # SGD优化器动量。
save_checkpoint: True                       # 是否保存检查点。
save_checkpoint_steps: 97                   # 两个检查点之间的迭代间隙。默认情况下，最后一个检查点将在最后一步迭代结束后保存。
keep_checkpoint_max: 30                     # 只保留最后一个keep_checkpoint_max检查点。
save_checkpoint_path: "./checkpoints"       # 检查点保存路径，相对于train.py。
```

## 数据集准备

- 您可以参考[快速入门](#quick-start)中的“生成数据集”自动生成数据集，也可以自行选择生成验证码数据集。

## 训练过程

- 在`default_config.yaml`中设置选项，包括学习率和网络超参数。单击[MindSpore加载数据集教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dataset_sample.html)，了解更多信息。

### 训练

- 在Ascend或GPU上运行`run_standalone_train.sh`进行WarpCTC模型的非分布式训练。

``` bash
bash run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_TARGET]
```

### 分布式训练

- 在Ascend上运行`run_distribute_train.sh`进行WarpCTC模型的分布式训练。

``` bash
bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

- 在GPU上运行`run_distribute_train_gpu.sh`进行WarpCTC模型的分布式训练。

``` bash
bash run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]
```

## 评估过程

### 评估

- 运行`run_eval.sh`进行评估。

``` bash
bash run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DEVICE_TARGET]
```

## 推理过程

### 导出MindIR

- 在本地导出

  ```shell
  python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

- 在ModelArts上导出

  ```python
  # (1) 上传你的代码到 s3 桶上
  # (2) 在ModelArts上创建训练任务
  # (3) 选择代码目录 /{path}/warpctc
  # (4) 选择启动文件 /{path}/warpctc/export.py
  # (5) 执行a或b
  #     a. 在 /{path}/warpctc/default_config.yaml 文件中设置参数
  #         1. 设置 ”enable_modelarts: True“
  #         2. 设置 “ckpt_file: ./{path}/*.ckpt”('ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
  #         3. 设置 ”file_name: warpctc“
  #         4. 设置 ”file_format：MINDIR“
  #     b. 在 网页上设置
  #         1. 添加 ”enable_modelarts=True“
  #         2. 添加 “ckpt_file=./{path}/*.ckpt”(('ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
  #         3. 添加 ”file_name=warpctc“
  #         4. 添加 ”file_format=MINDIR“
  # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径(这一步不起作用，但必须要有)
  # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
  # (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
  # (10) 创建训练作业
  # 你将在{Output file path}下看到 'warpctc.mindir'文件
  ```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前仅支持batch_size为1的推理。
采用mindir+bin方式进行推理，其中bin为预处理完的图片的二进制文件。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DATA_PATH` 为必填项，数据格式为bin的路径。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'Accuracy':0.952
```

## 模型描述

### 性能

#### 训练性能

| 参数                 | Ascend 910                                    |   GPU |
| -------------------------- | --------------------------------------------- |---------------------------------- |
| 模型版本              | v1.0                                          | v1.0 |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8   | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24核，内存： 128G
| 上传日期              | 2021-07-05                   | 2021-07-05 |
| MindSpore版本          | 1.3.0                                         | 1.3.0       |
| 数据集                    | Captcha                                       | Captcha |
| 训练参数        | epoch=30, steps per epoch=98, batch_size = 64 | epoch=30, steps per epoch=98, batch_size = 64  |
| 优化器                  | SGD                                           | SGD |
| 损失函数              | CTCLoss                                       | CTCLoss |
| 输出                    | 概率                                   | 概率 |
| 损失                       | 0.0000157                                     | 0.0000246  |
| 速度                      | 980毫秒/步（8卡）                             | 150毫秒/步（8卡）|
| 总时长                 | 30分钟                                       | 5分钟|
| 参数(M)             | 2.75                                          | 2.75 |
| 微调检查点 | 20.3M (.ckpt文件)                            | 20.3M (.ckpt文件) |
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc) | [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc) |

#### 评估性能

| 参数          | WarpCTC                     |
| ------------------- | --------------------------- |
| 模型版本       | V1.0                        |
| 资源            |Ascend 910；系统 Euler2.8                 |
| 上传日期       | 2021-07-05 |
| MindSpore版本   | 1.3.0                 |
| 数据集             | Captcha                     |
| batch_size          | 64                          |
| 输出             | ACC                         |
| 准确率            | 99.0%                       |
| 推理模型 | 20.3M (.ckpt文件)          |

### 推理性能

| 参数            | Ascend                      |
| ------------- | ----------------------------|
| 模型版本        | WarpCTC                     |
| 资源           | Ascend 310；系统 CentOS 3.10 |
| 上传日期        | 2021-05-24                  |
| MindSpore版本  | 1.2.0                       |
| 数据集          | Captcha                     |
| batch_size     | 1                          |
| 输出            | Accuracy                   |
| 准确率          | Accuracy=0.952             |
| 推理模型        | 40.6M（.ckpt文件）           |

## 随机情况说明

在dataset.py中设置“create_dataset”函数内的种子。使用train.py中的随机种子进行权重初始化。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
