# 目录

- [目录](#目录)
- [Tiny-DarkNet描述](#tiny-darknet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本描述](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
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
- [ModelZoo主页](#modelzoo主页)

# [Tiny-DarkNet描述](#目录)

Tiny-DarkNet是Joseph Chet Redmon等人提出的一个16层的针对于经典的图像分类数据集ImageNet所进行的图像分类网络模型。 Tiny-DarkNet作为作者为了满足用户对较小模型规模的需求而尽量降低模型的大小设计的简易版本的Darknet，具有优于AlexNet和SqueezeNet的图像分类能力，同时其只使用少于它们的模型参数。为了减少模型的规模，该Tiny-DarkNet网络没有使用全连接层，仅由卷积层、最大池化层、平均池化层组成。

更多Tiny-DarkNet详细信息可以参考[官方介绍](https://pjreddie.com/darknet/tiny-darknet/)

# [模型架构](#目录)

具体而言, Tiny-DarkNet网络由**1×1 conv**, **3×3 conv**, **2×2 max**和全局平均池化层组成，这些模块相互组成将输入的图片转换成一个**1×1000**的向量。

# [数据集](#目录)

以下将介绍模型中使用数据集以及其出处：
<!-- Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below. -->

<!-- Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)  -->

<!-- Dataset used ImageNet can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size: 125G, 1250k colorful images in 1000 classes
  - Train: 120G, 1200k images
  - Test: 5G, 50k images
- Data format: RGB images.
  - Note: Data will be processed in src/dataset.py  -->

所使用的数据集可参考[论文](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- 数据集规模：125G，1250k张分别属于1000个类的彩色图像
    - 训练集: 120G,1200k张图片
    - 测试集: 5G, 50k张图片
- 数据格式: RGB格式图片
    - 注意: 数据将会被 src/dataset.py 中的函数进行处理

<!-- # [Features](#contents)

## [Distributed](#contents)

<!-- 不同的机器有同一个模型的多个副本，每个机器分配到不同的数据，然后将所有机器的计算结果按照某种方式合并 -->

<!-- 在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。[分布式并行训练](<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training.html>)，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。本模型使用了mindspore提供的自动并行模式AUTO_PARALLEL：该方法是融合了数据并行、模型并行及混合并行的1种分布式并行模式，可以自动建立代价模型，找到训练时间较短的并行策略，为用户选择1种并行模式。 -->

# [环境要求](#目录)

- 硬件（Ascend/CPU/GPU）
    - 请准备具有Ascend/CPU处理器/GPU的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多的信息请访问以下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [快速入门](#目录)

根据官方网站成功安装MindSpore以后，可以按照以下步骤进行训练和测试模型：

- 在Ascend资源上运行：

  ```python
  # 单卡训练
  bash ./scripts/run_standalone_train.sh 0

  # 分布式训练
  bash ./scripts/run_distribute_train.sh /{path}/*.json

  # 评估
  python eval.py > eval.log 2>&1 &
  OR
  bash ./script/run_eval.sh
  ```

  进行并行训练时, 需要提前创建JSON格式的hccl配置文件 [RANK_TABLE_FILE]。

  请按照以下链接的指导进行设置:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

- running on GPU with gpu default parameters

  ```python
  # GPU单卡训练示例
  python train.py  \
  --config_path=./config/imagenet_config_gpu.yaml \
  --dataset_name=imagenet --train_data_dir=../dataset/imagenet_original/train --device_target=GPU
  OR
  cd scripts
  bash run_distribute_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10 | imagenet]

  # GPU多卡训练示例
  export RANK_SIZE=8
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout  \
  python train.py  \
  --config_path=./config/imagenet_config_gpu.yaml \
  --dataset_name=imagenet \
  --train_data_dir=../dataset/imagenet_original/train \
  --device_target=GPU
  OR
  bash scripts/run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR] [cifar10 | imagenet]

  # GPU评估示例
  python eval.py -device_target=GPU --val_data_dir=../dataset/imagenet_original/val --dataset_name=imagenet --config_path=./config/imagenet_config_gpu.yaml \
  --checkpoint_path=$PATH2
  OR
  bash scripts/run_train_gpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

- 在ModelArts上运行
      如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/)

    - 在ModelArt上使用8卡训练

      ```python
      # (1) 上传你的代码到 s3 桶上
      # (2) 在ModelArts上创建训练任务
      # (3) 选择代码目录 /{path}/tinydarknet
      # (4) 选择启动文件 /{path}/tinydarknet/train.py
      # (5) 执行a或b
      #     a. 在 /{path}/tinydarknet/imagenet_config.yaml 文件中设置参数
      #         1. 设置 ”batch_size: 64“ (非必须)
      #         2. 设置 ”enable_modelarts: True“
      #         3. 如果数据采用zip格式压缩包的形式上传，设置 ”modelarts_dataset_unzip_name: {filenmae}"
      #     b. 在 网页上设置
      #         1. 添加 ”batch_size=True“
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
      # (3) 选择代码目录 /{path}/tinydarknet
      # (4) 选择启动文件 /{path}/tinydarknet/eval.py
      # (5) 执行a或b
      #     a. 在 /path/tinydarknet 下的imagenet_config.yaml 文件中设置参数
      #         1. 设置 ”enable_modelarts: True“
      #         2. 设置 “checkpoint_path: {checkpoint_path}”({checkpoint_path}表示待评估的 权重文件 相对于 eval.py 的路径,权重文件须包含在代码目录下。)
      #         3. 如果数据采用zip格式压缩包的形式上传，设置 ”modelarts_dataset_unzip_name: {filenmae}"
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

更多的细节请参考具体的script文件

# [脚本描述](#目录)

## [脚本及样例代码](#目录)

```bash

├── tinydarknet
├── README.md                           // Tiny-Darknet英文说明
    ├── README_CN.md                    // Tiny-Darknet中文说明
    ├── ascend310_infer                 // 用于310推理
    ├── config
        ├── imagenet_config.yaml        // imagenet参数配置
        ├── imagenet_config_gpu.yaml    // imagenet参数配置
        ├── cifar10_config.yaml         // cifar10参数配置
        ├── cifar10_config_gpu.yaml     // cifar10参数配置
    ├── scripts
        ├── run_standalone_train.sh     // Ascend单卡训练shell脚本
        ├── run_standalone_train_gpu.sh // GPU单卡训练shell脚本  
        ├── run_distribute_train.sh     // Ascend分布式训练shell脚本
        ├── run_distribute_train_gpu.sh // GPU分布式训练shell脚本
        ├── run_train_cpu.sh            // CPU训练shell脚本
        ├── run_eval.sh                 // Ascend评估shell脚本
        ├── run_eval_cpu.sh             // CPU评估shell脚本
        ├── run_eval_gpu.sh             // GPU评估shell脚本
        └── run_infer_310.sh            // Ascend310推理shell脚本
    ├── src
        ├── lr_scheduler                // 学习率策略
            ├── __init__.py             // 初始化文件
            ├── linear_warmup.py        // linear_warmup策略
            ├── warmup_cosine_annealing_lr.py     // warmup_cosine_annealing_lr策略
            └── warmup_step_lr.py       // warmup_step_lr策略
        ├── model_utils
            ├── config.py               // 解析 *.yaml 参数配置文件
            ├── device_adapter.py       // 区分本地/ModelArts训练
            ├── local_adapter.py        // 本地训练获取相关环境变量
            └── moxing_adapter.py       // ModelArts训练获取相关环境变量、交换数据
        ├── dataset.py                  // 创建数据集
        ├── CrossEntropySmooth.py       // 损失函数
        └── tinydarknet.py              // Tiny-Darknet网络结构
    ├── train.py                        // 训练脚本
    ├── eval.py                         // 评估脚本
    ├── export.py                       // 导出checkpoint文件
    ├── mindspore_hub_conf.py           // hub配置文件
    └── postprocess.py                  // 310推理后处理脚本

```

## [脚本参数](#目录)

训练和测试的参数可在 `imagenet_config.yaml` 中进行设置

- Tiny-Darknet的配置文件(仅列出部分参数)

  ```python
  pre_trained: False      # 是否载入预训练模型
  num_classes: 1000       # 数据集中类的数量
  lr_init: 0.1            # 初始学习率
  batch_size: 128         # 训练的batch_size
  epoch_size: 500         # 总共的训练epoch
  momentum: 0.9           # 动量
  weight_decay: 1e-4      # 权重衰减率
  image_height: 224       # 输入图像的高度
  image_width: 224        # 输入图像的宽度
  train_data_dir: './ImageNet_Original/train/'    # 训练数据集的绝对路径
  val_data_dir: './ImageNet_Original/val/'  # 评估数据集的绝对路径
  device_target: 'Ascend' # 程序运行的设备
  keep_checkpoint_max: 10 # 仅仅保持最新的keep_checkpoint_max个checkpoint文件
  checkpoint_path: '/train_tinydarknet.ckpt'  # 保存checkpoint文件的绝对路径
  onnx_filename: 'tinydarknet.onnx' # 用于export.py 文件中的onnx模型的文件名
  air_filename: 'tinydarknet.air'   # 用于export.py 文件中的air模型的文件名
  lr_scheduler: 'exponential'     # 学习率策略
  lr_epochs: [70, 140, 210, 280]  # 学习率进行变化的epoch数
  lr_gamma: 0.3            # lr_scheduler为exponential时的学习率衰减因子
  eta_min: 0.0             # cosine_annealing策略中的eta_min
  T_max: 150               # cosine_annealing策略中的T-max
  warmup_epochs: 0         # 热启动的epoch数
  is_dynamic_loss_scale: 0 # 动态损失尺度
  loss_scale: 1024         # 损失尺度
  label_smooth_factor: 0.1 # 训练标签平滑因子
  use_label_smooth: True   # 是否采用训练标签平滑
  ```

更多的细节, 请参考`imagenet_config.yaml`.

## [训练过程](#目录)

### [单机训练](#目录)

- 在Ascend资源上运行：

  ```python
  bash ./scripts/run_standalone_train.sh [DEVICE_ID]
  ```

  上述的命令将运行在后台中，可以通过 `train.log` 文件查看运行结果.

  训练完成后,默认情况下,可在script文件夹下得到一些checkpoint文件. 训练的损失值将以如下的形式展示:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

  ```python
  # grep "loss is " train.log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

  模型checkpoint文件将会保存在当前文件夹下.
  <!-- The model checkpoint will be saved in the current directory.  -->

- 在GPU资源上运行：

  ```python
  cd scripts
  bash run_standalone_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

  上述的命令将运行在后台中，可以通过 `train_single_gpu/train.log` 文件查看运行结果.

  训练完成后,默认情况下,可在script文件夹下得到一些checkpoint文件. 训练的损失值将以如下的形式展示:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

  ```python
  # grep "loss is " train.log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

- 在CPU资源上运行：

  ```python
  bash scripts/run_train_cpu.sh [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

### [分布式训练](#目录)

- 在Ascend资源上运行：

  ```python
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE]
  ```

  上述的脚本命令将在后台中进行分布式训练，可以通过`distribute_train/nohup.out`文件查看运行结果. 训练的损失值将以如下的形式展示:

  ```python
  # grep "result: " distribute_train/nohup.out
  epoch: 498 step: 1251, loss is 2.7825122
  epoch time: 200066.210 ms, per step time: 159.925 ms
  epoch: 499 step: 1251, loss is 2.799798
  epoch time: 199098.258 ms, per step time: 159.151 ms
  epoch: 500 step: 1251, loss is 2.8718748
  epoch time: 197784.661 ms, per step time: 158.101 ms
  ...
  ```

- 在GPU资源上运行：

  ```python
  bash scripts/run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

  上述的脚本命令将在后台中进行分布式训练，可以通过`distribute_train_gpu/nohup.out`文件查看运行结果. 训练的损失值将以如下的形式展示:

  ```python
  # grep "result: " distribute_train_gpu/nohup.out
  epoch: 498 step: 1251, loss is 2.7825122
  epoch time: 200066.210 ms, per step time: 159.925 ms
  epoch: 499 step: 1251, loss is 2.799798
  epoch time: 199098.258 ms, per step time: 159.151 ms
  epoch: 500 step: 1251, loss is 2.8718748
  epoch time: 197784.661 ms, per step time: 158.101 ms
  ...
  ```

## [评估过程](#目录)

### [评估](#目录)

- 在Ascend资源上进行评估:

  在运行如下命令前,请确认用于评估的checkpoint文件的路径.checkpoint文件须包含在tinydarknet文件夹内.请将checkpoint路径设置为相对于 eval.py文件 的路径,例如:"./ckpts/train_tinydarknet.ckpt"(ckpts 与 eval.py 同级).

  ```python
  python eval.py > eval.log 2>&1 &  
  OR
  bash scripts/run_eval.sh
  ```

  上述的python命令将运行在后台中，可以通过"eval.log"文件查看结果. 测试数据集的准确率将如下面所列:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

  请注意在并行训练后,测试请将checkpoint_path设置为最后保存的checkpoint文件的路径,准确率将如下面所列:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

- 在GPU资源上进行评估:

  在运行如下命令前,请确认用于评估的checkpoint文件的路径.checkpoint文件须包含在tinydarknet文件夹内.请将checkpoint路径设置为相对于 eval.py文件 的路径,例如:"./ckpts/train_tinydarknet.ckpt"(ckpts 与 eval.py 同级).

  ```python
  bash scripts/run_train_gpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

  上述的python命令将运行在后台中，可以通过"eval.log"文件查看结果. 测试数据集的准确率将如下面所列:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5896033653846153, 'top_5_accuracy': 0.8176482371794872}
  ```

  请注意在并行训练后,测试请将checkpoint_path设置为最后保存的checkpoint文件的路径,准确率将如下面所列:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5896033653846153, 'top_5_accuracy': 0.8176482371794872}
  ```

- 在CPU资源上进行评估

  在运行如下命令前,请确认用于评估的checkpoint文件的路径.checkpoint文件须包含在tinydarknet文件夹内.请将checkpoint路径设置为相对于 eval.py文件 的路径,例如:"./ckpts/train_tinydarknet.ckpt"(ckpts 与 eval.py 同级).

  ```python
  bash scripts/run_eval_cpu.sh [VAL_DATA_DIR] [imagenet|cifar10] [CHECKPOINT_PATH]
  ```

  可以通过"eval.log"文件查看结果. 测试数据集的准确率将如下面所列:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_5_accuracy': 1.0, 'top_1_accuracy': 0.9829727564102564}
  ```

## 推理过程

### 导出MindIR

- 在本地导出

```shell
python export.py --dataset [DATASET] --file_name [FILE_NAME] --file_format [EXPORT_FORMAT]
```

- 在ModelArts上导出

  ```python
  # (1) 上传你的代码到 s3 桶上
  # (2) 在ModelArts上创建训练任务
  # (3) 选择代码目录 /{path}/tinydarknet
  # (4) 选择启动文件 /{path}/tinydarknet/export.py
  # (5) 执行a或b
  #     a. 在 /{path}/tinydarknet/default_config.yaml 文件中设置参数
  #         1. 设置 ”enable_modelarts: True“
  #         2. 设置 “checkpoint_path: ./{path}/*.ckpt”('checkpoint_path' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
  #         3. 设置 ”file_name: tinydarknet“
  #         4. 设置 ”file_format：MINDIR“
  #     b. 在 网页上设置
  #         1. 添加 ”enable_modelarts=True“
  #         2. 添加 “checkpoint_path=./{path}/*.ckpt”(('checkpoint_path' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
  #         3. 添加 ”file_name=tinydarknet“
  #         4. 添加 ”file_format=MINDIR“
  # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径(这一步不起作用，但必须要有)
  # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
  # (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
  # (10) 创建训练作业
  # 你将在{Output file path}下看到 'tinydarknet.mindir'文件
  ```

参数没有ckpt_file选项，ckpt文件请按照`imagenet_config.yaml`中参数`checkpoint_path`的路径存放。
`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"].

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前仅支持batch_size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
```

- `LABEL_PATH` label.txt存放的路径，写一个py脚本对数据集下的类别名进行排序，对类别下的文件名和类别排序值做映射，例如[文件名:排序值]，将映射结果写到labe.txt文件中。
- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。注意目前仅支持CPU模式。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
'top_1_accuracy': 59.07%, 'top_5_accuracy': 81.73%
```

# [模型描述](#目录)

## [性能](#目录)

### [训练性能](#目录)

#### Tinydarknet on ImageNet 2012

| 参数                       | Ascend                                                      | GPU                                                 |
| -------------------------- | ------------------------------------------------------------| ----------------------------------------------------|
| 模型版本                   | V1                                                          | V1                                                  |
| 资源                       | Ascend 910；CPU 2.60GHz，56cores；内存 314G；系统 Euler2.8  | PCIE V100-32G                                       |
| 上传日期                   | 2020/12/22                                                  | 2021/07/15                                          |
| MindSpore版本              | 1.1.0                                                       | 1.3.0                                               |
| 数据集                     | 1200k张图片                                                 | 1200k张图片                                         |
| 训练参数                   | epoch=500, steps=1251, batch_size=128, lr=0.1               | epoch=500, steps=1251, batch_size = 128, lr=0.005   |
| 优化器                     | Momentum                                                    | Momentum                                            |
| 损失函数                   | Softmax Cross Entropy                                       | Softmax Cross Entropy                               |
| 速度                       | 8卡: 104 ms/step                                            | 8卡: 255 ms/step                                    |
| 总时间                     | 8卡: 17.8小时                                               | 8卡: 46.9小时                                       |
| 参数(M)                    | 4.0;                                                        | 4.0;                                              |
| 脚本                       | [Tiny-Darknet脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/tinydarknet)

### [评估性能](#目录)

| 参数                | Ascend                            | GPU                               |
| ------------------- | ----------------------------------| ----------------------------------|
| 模型版本            | V1                                | V1                                |
| 资源                |  Ascend 910；系统 Euler2.8        | NV SMX2 V100-32G                  |
| 上传日期            | 2020/12/22                        | 2021/7/15                         |
| MindSpore版本       | 1.1.0                             | 1.3.0                             |
| 数据集              | 200k张图片                        | 200k张图片                        |
| batch_size          | 128                               | 128                               |
| 输出                | 分类概率                          | 分类概率                          |
| 准确率              | 8卡 Top-1: 58.7%; Top-5: 81.7%    | 8卡 Top-1: 58.9%; Top-5: 81.7%    |
| 推理模型            | 11.6M (.ckpt文件)                 | 10.06M (.ckpt文件)                |

### [推理性能](#目录)

| 参数            | Ascend                     |
| -------------- | ---------------------------|
| 模型版本        | TinyDarknet                 |
| 资源           | Ascend 310；系统 Euler2.8 |
| 上传日期        | 2021-05-29                  |
| MindSpore版本  | 1.2.0                       |
| 数据集          | ImageNet                   |
| batch_size     | 1                          |
| 输出            | Accuracy                   |
| 准确率          | Top-1: 59.07% Top-5: 81.73%  |
| 推理模型        | 10.3M（.ckpt文件）            |

# [ModelZoo主页](#目录)

 请参考官方[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
