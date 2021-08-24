# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DeepSpeech2介绍](#DeepSpeech2介绍)
- [网络模型结构](#网络模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [文件说明和运行说明](#文件说明和运行说明)
    - [代码目录结构说明](#代码目录结构说明)
    - [模型参数](#模型参数)
    - [训练和推理过程](#训练和推理过程)
    - [Export](#Export)
- [性能](#性能)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

# [DeepSpeech2介绍](#contents)

DeepSpeech2是一个使用 CTC 损失训练的语音识别模型。它用神经网络取代了整个手工设计的管道，可以处理各种各样的语音，包括嘈杂的环境、口音和不同的语言。

[论文](https://arxiv.org/pdf/1512.02595v1.pdf): Amodei, Dario, et al. Deep speech 2: End-to-end speech recognition in english and mandarin.

# [网络模型结构](#contents)

模型包括:

- 两个卷积层:
    - 通道数为 32，内核大小为  41, 11 ，步长为  2, 2
    - 通道数为 32，内核大小为  41, 11 ，步长为  2, 1
- 五个双向 LSTM 层（大小为 1024）
- 一个投影层【大小为字符数加 1（为CTC空白符号)，29】

# [数据集](#contents)

可以基于论文中提到的数据集或在相关领域/网络架构中广泛使用的数据集运行脚本。在下面的部分中，我们将介绍如何使用下面的相关数据集运行脚本。

使用的数据集为: [LibriSpeech](<http://www.openslr.org/12>)

- 训练集：
    - train-clean-100: [6.3G] (100小时的无噪音演讲训练集)
    - train-clean-360.tar.gz [23G] (360小时的无噪音演讲训练集)
    - train-other-500.tar.gz [30G] (500小时的有噪音演讲训练集)
- 验证集：
    - dev-clean.tar.gz [337M] (无噪音)
    - dev-other.tar.gz [314M] (有噪音)  
- 测试集:
    - test-clean.tar.gz [346M] (测试集, 无噪音)
    - test-other.tar.gz [328M] (测试集, 有噪音)
- 数据格式：wav 和 txt 文件
    - 注意：数据需要通过librispeech.py进行处理

# [环境要求](#contents)

- 硬件（GPU）
    - GPU处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [文件说明和运行说明](#contents)

## [代码目录结构说明](#contents)

```path
.
├── audio
    ├── deepspeech2
        ├── scripts
        │   ├──run_distribute_train_gpu.sh // gpu8卡训练脚本
        │   ├──run_eval_cpu.sh             // cpu推理脚本
        │   ├──run_eval_gpu.sh             // gpu推理脚本
        │   ├──run_standalone_train_cpu.sh // cpu单卡训练脚本
        │   └──run_standalone_train_gpu.sh // gpu单卡训练脚本
        ├── train.py                       // 训练文件
        ├── eval.py                        // 推理文件
        ├── export.py                      // 将mindspore模型转换为mindir模型
        ├── labels.json                    // 可能映射到的字符
        ├── README.md                      // DeepSpeech2相关描述
        ├── deepspeech_pytorch             //
            ├──decoder.py                  // 来自第三方代码的解码器（MIT 许可证）
        ├── src
            ├──__init__.py
            ├──DeepSpeech.py               // DeepSpeech2网络架构
            ├──dataset.py                  // 数据处理
            ├──config.py                   // DeepSpeech配置文件
            ├──lr_generator.py             // 产生学习率
            ├──greedydecoder.py            // 修改Mindspore代码的greedydecoder
            └──callback.py                 // 回调以监控训练
```

## [模型参数](#contents)

训练和推理的相关参数在`config.py`文件

```text
训练相关参数
    epochs                       训练的epoch数量，默认为70
```

```text
数据处理相关参数
    train_manifest               用于训练的数据文件路径，默认为 'data/libri_train_manifest.csv'
    val_manifest                 用于测试的数据文件路径，默认为 'data/libri_val_manifest.csv'
    batch_size                   批处理大小，默认为8
    labels_path                  模型输出的token json 路径, 默认为 "./labels.json"
    sample_rate                  数据特征的采样率，默认为16000
    window_size                  频谱图生成的窗口大小（秒），默认为0.02
    window_stride                频谱图生成的窗口步长（秒），默认为0.01
    window                       频谱图生成的窗口类型，默认为 'hamming'
    speed_volume_perturb         使用随机速度和增益扰动，默认为False，当前模型中未使用
    spec_augment                 在MEL谱图上使用简单的光谱增强，默认为False，当前模型中未使用
    noise_dir                    注入噪音到音频。默认为noise Inject未添加，默认为''，当前模型中未使用
    noise_prob                   每个样本加噪声的概率，默认为0.4，当前模型中未使用
    noise_min                    样本的最小噪音水平，(1.0意味着所有的噪声，不是原始信号)，默认是0.0，当前模型中未使用
    noise_max                    样本的最大噪音水平。最大值为1.0，默认值为0.5，当前模型中未使用
```

```text
模型相关参数
    rnn_type                     模型中使用的RNN类型，默认为'LSTM'，当前只支持LSTM
    hidden_size                  RNN层的隐藏大小，默认为1024
    hidden_layers                RNN层的数量，默认为5
    lookahead_context            查看上下文，默认值是20，当前模型中未使用
```

```text
优化器相关参数
    learning_rate                初始化学习率，默认为3e-4
    learning_anneal              对每个epoch之后的学习率进行退火，默认为1.1
    weight_decay                 权重衰减，默认为1e-5
    momentum                     动量，默认为0.9
    eps                          Adam eps，默认为1e-8
    betas                        Adam betas，默认为(0.9, 0.999)
    loss_scale                   损失规模，默认是1024
```

```text
checkpoint相关参数
    ckpt_file_name_prefix        ckpt文件的名称前缀，默认为'DeepSpeech'
    ckpt_path                    ckpt文件的保存路径，默认为'checkpoints'
    keep_checkpoint_max          ckpt文件的最大数量限制，删除旧的检查点，默认是10
```

# [训练和推理过程](#contents)

## 训练

```text
运行: train.py   [--use_pretrained USE_PRETRAINED]
                 [--pre_trained_model_path PRE_TRAINED_MODEL_PATH]
                 [--is_distributed IS_DISTRIBUTED]
                 [--bidirectional BIDIRECTIONAL]
                 [--device_target DEVICE_TARGET]
参数:
    --pre_trained_model_path    预先训练的模型文件路径，默认为''
    --is_distributed            多卡训练，默认为False
    --bidirectional             是否使用双向RNN，默认为True，目前只实现了双向模型
    --device_target             运行代码的设备："GPU" | “CPU”，默认为"GPU"
```

## 推理

```text
运行: eval.py   [--bidirectional BIDIRECTIONAL]
                [--pretrain_ckpt PRETRAIN_CKPT]
                [--device_target DEVICE_TARGET]

参数:
    --bidirectional              是否使用双向RNN，默认为True。 目前只实现了双向模型
    --pretrain_ckpt              checkpoint的文件路径, 默认为''
    --device_target              运行代码的设备："GPU" | “CPU”，默认为"GPU"
```

在训练之前，应该处理数据集，使用[SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch)中的脚本来处理数据。
[SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch)中的脚本文件将自动下载数据集并进行处理。
流程结束后，数据目录结构如下：

```path
    .
    ├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── val
    │  │    ├─ wav
    │  │    └─ txt
    │  ├── test_clean  
    │  │    ├─ wav
    │  │    └─ txt  
    │  └── test_other
    │       ├─ wav
    │       └─ txt
    └─ libri_test_clean_manifest.csv, libri_test_other_manifest.csv, libri_train_manifest.csv, libri_val_manifest.csv
```

三个*.csv文件存放的是对应数据的绝对路径，得到3个csv文件后，修改src/config.py中的配置。
对于训练配置, train_manifest应该配置为`libri_train_manifest.csv`的路径，对于 eval 配置，应该配置为 `libri_test_other_manifest.csv` 或 `libri_train_manifest.csv`，具体取决于评估的数据集。

```shell
...
训练配置
"DataConfig":{
     train_manifest:'path_to_csv/libri_train_manifest.csv'
}

评估配置
"DataConfig":{
     train_manifest:'path_to_csv/libri_test_clean_manifest.csv'
}

```

训练之前，需要安装`librosa` and `Levenshtein`
通过官网安装MindSpore并完成数据集处理后，可以开始训练如下：

```shell

# gpu单卡训练
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID]

# cpu单卡训练
bash ./scripts/run_standalone_train_cpu.sh

# gpu多卡训练
bash ./scripts/run_distribute_train_gpu.sh

```

进行模型评估需要注意的是：目前在运行脚本之前只支持greedy decoder，可以从[SeanNaren](https://github.com/SeanNaren/deepspeech.pytorch)下载解码器并将
deepspeech_pytorch文件放入deepspeech2目录， 之后文件目录将显示为[Script and Sample Code]

```shell

# cpu评估
bash ./scripts/run_eval_cpu.sh [PATH_CHECKPOINT]

# gpu评估
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [PATH_CHECKPOINT]

```

## [Export](#contents)

```bash
python export.py --pre_trained_model_path='ckpt_path'
```

# [性能](#contents)

## [训练和测试性能分析](#contents)

### 训练性能

| 参数                 | DeepSpeech                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | NV SMX2 V100-32G              |
| 更新日期              | 12/29/2020 (month/day/year)                                    |
| MindSpore版本           | 1.0.0                                                          |
| 数据集                    | LibriSpeech                                                 |
| 训练参数       | 2p, epoch=70, steps=5144 * epoch, batch_size = 20, lr=3e-4   |
| 优化器                  | Adam                                                           |
| 损失函数              | CTCLoss                                |
| 输出                    | 概率值                                                    |
| 损失值                       | 0.2-0.7                                                        |
| 运行速度                      | 2p 2.139s/step                                   |
| 训练总时间       | 2p: around 1 week;                                  |
| Checkpoint文件大小                 | 991M (.ckpt file)                                              |
| 代码                   | [DeepSpeech script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/deepspeech2) |

### Inference Performance

| 参数                 | DeepSpeech                                                       |
| -------------------------- | ----------------------------------------------------------------|
| 资源                   | NV SMX2 V100-32G                   |
| 更新日期              | 12/29/2020 (month/day/year)                                 |
| MindSpore版本          | 1.0.0                                                           |
| 数据集                    | LibriSpeech                         |
| 批处理大小                 | 20                                                               |
| 输出                    | 概率值                       |
| 精确度(无噪声)       | 2p: WER: 9.902  CER: 3.317  8p: WER: 11.593  CER: 3.907|
| 精确度(有噪声)      | 2p: WER: 28.693 CER: 12.473 8p: WER: 31.397  CER: 13.696|
| 模型大小        | 330M (.mindir file)                                              |

# [ModelZoo主页](#contents)

 [ModelZoo主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
