# 目录

<!-- TOC -->

- [目录](#目录)
    - [DeepFM概述](#deepfm概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
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
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## DeepFM概述

要想在推荐系统中实现最大点击率，学习用户行为背后复杂的特性交互十分重要。虽然已在这一领域取得很大进展，但高阶交互和低阶交互的方法差异明显，亟需专业的特征工程。本论文中,我们将会展示高阶和低阶交互的端到端学习模型的推导。本论文提出的模型DeepFM，结合了推荐系统中因子分解机和新神经网络架构中的深度特征学习。

[论文](https://arxiv.org/abs/1703.04247):  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

## 模型架构

DeepFM由两部分组成。FM部分是一个因子分解机，用于学习推荐的特征交互；深度学习部分是一个前馈神经网络，用于学习高阶特征交互。
FM和深度学习部分拥有相同的输入原样特征向量，让DeepFM能从输入原样特征中同时学习低阶和高阶特征交互。

## 数据集

- [1] A dataset used in  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017.

## 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 数据集预处理
  '''bash
  #下载数据集
  #请参考[1]获得下载链接
  mkdir -p data/origin_data && cd data/origin_data
  wget DATA_LINK
  tar -zxvf dac.tar.gz

  #数据集预处理脚步执行
  python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
  '''

- Ascend处理器环境运行

  ```训练示例
  # 运行训练示例
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json

  # 运行评估示例
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  在分布式训练中，JSON格式的HCCL配置文件需要提前创建。

  具体操作，参见：

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- 在GPU上运行

  如在GPU上运行,请配置文件src/config.py中的`device_target`从 `Ascend`改为`GPU`。

  ```训练示例
  # 运行训练示例
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh 8 /dataset_path

  # 运行评估示例
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='GPU' > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path/deepfm.ckpt
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "train_epochs: 5"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "train_epochs: 5"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/deepfm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "train_epochs: 5"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "train_epochs: 5"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/deepfm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./deepfm/deepfm_trained.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./deepfm/deepfm_trained.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/deepfm"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='deepfm'"
    #          在 base_config.yaml 文件中设置 "file_format='AIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='deepfm'"
    #          在网页上设置 "file_format='AIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/deepfm"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## 脚本说明

## 脚本和样例代码

```deepfm
.
└─deepfm
  ├─README.md
  ├─mindspore_hub_conf.md             # mindspore hub配置
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend处理器或GPU上进行单机训练(单卡)
    ├─run_distribute_train.sh         # 在Ascend处理器上进行分布式训练(8卡)
    ├─run_distribute_train_gpu.sh     # 在GPU上进行分布式训练(8卡)
    └─run_eval.sh                     # 在Ascend处理器或GPU上进行评估
  ├─src
    ├─__init__.py                     # python init文件
    ├─config.py                       # 参数配置
    ├─callback.py                     # 定义回调功能
    ├─deepfm.py                       # DeepFM网络
    ├─dataset.py                      # 创建DeepFM数据集
  ├─eval.py                           # 评估网络
  └─train.py                          # 训练网络
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 训练参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path
  --ckpt_path CKPT_PATH
                        Checkpoint path
  --eval_file_name EVAL_FILE_NAME
                        Auc log file path. Default: "./auc.log"
  --loss_file_name LOSS_FILE_NAME
                        Loss log file path. Default: "./loss.log"
  --do_eval DO_EVAL     Do evaluation or not. Default: True
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

- 评估参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

## 训练过程

### 训练

- Ascend处理器上运行

  ```运行命令
  python trin.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  上述python命令将在后台运行,您可以通过`ms_log/output.log`文件查看结果。

  训练结束后, 您可在默认文件夹`./checkpoint`中找到检查点文件。损失值保存在loss.log文件中。

  ```运行结果
  2020-05-27 15:26:29 epoch: 1 step: 41257, loss is 0.498953253030777
  2020-05-27 15:32:32 epoch: 2 step: 41257, loss is 0.45545706152915955
  ...
  ```

  模型检查点将会储存在当前路径。

- GPU上运行
  待运行。

### 分布式训练

- Ascend处理器上运行

  ```运行命令
  bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json
  ```

  上述shell脚本将在后台运行分布式训练。请在`log[X]/output.log`文件中查看结果。损失值保存在loss.log文件中。

- GPU上运行
  待运行。

## 评估过程

### 评估

- Ascend处理器上运行时评估数据集

  在运行以下命令之前，请检查用于评估的检查点路径。

  ```命令
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  上述python命令将在后台运行，请在eval_output.log路径下查看结果。准确率保存在auc.log文件中。

  ```结果
  {'result': {'AUC': 0.8057789065281104, 'eval_time': 35.64779996871948}}
  ```

- 在GPU运行时评估数据集
  待运行。

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
auc : 0.8057789065281104
```

## 模型描述

## 性能

### 评估性能

| 参数                    | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | DeepFM                                                      | 待运行                  |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | 待运行                  |
| 上传日期              | 2021-07-05                                 | 待运行                 |
| MindSpore版本          | 1.3.0                                                 | 待运行                  |
| 数据集                    | [1]                                                         | 待运行                 |
| 训练参数        | epoch=15, batch_size=16000, lr=1e-5                          | 待运行                  |
| 优化器                  | Adam                                                        | 待运行                 |
| 损失函数              | Sigmoid Cross Entropy With Logits                           | 待运行                  |
| 输出                    | 准确率                                                    | 待运行                 |
| 损失                       | 0.45                                                        | 待运行                 |
| 速度| 单卡：21毫秒/步;                                          | 待运行                  |
| 总时长| 单卡：90 分钟;                                               | 待运行                 |
| 参数(M)             | 16.5                                                        | 待运行                  |
| 微调检查点 | 190M (.ckpt 文件)                                           | 待运行                  |
| 脚本                    | [DeepFM脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm) | 待运行                  |

### 推理性能

| 参数          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | DeepFM                      | 待运行                       |
| 资源            | Ascend 910；系统 Euler2.8                  | 待运行                       |
| 上传日期       | 2021-07-05 | 待运行                       |
| MindSpore版本   | 1.3.0                 | 待运行                       |
| 数据集             | [1]                         | 待运行                       |
| batch_size          | 1000                        | 待运行                      |
| 输出             | 准确率                    | 待运行                       |
| 准确率| 单卡：80.55%;                |待运行                       |
| 推理模型 | 190M (.ckpt文件)           | 待运行                       |

## 随机情况说明

在train.py.中训练之前设置随机种子。

## ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。  
