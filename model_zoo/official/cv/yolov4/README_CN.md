# 目录

- [目录](#目录)
- [YOLOv4说明](#yolov4说明)
- [模型架构](#模型架构)
- [预训练模型](#预训练模型)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
        - [迁移学习](#迁移学习)
    - [评估过程](#评估过程)
        - [验证](#验证)
        - [Test-dev](#test-dev)
    - [转换过程](#转换过程)
        - [转换](#转换)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [YOLOv4说明](#目录)

YOLOv4作为先进的检测器，它比所有可用的替代检测器更快（FPS）并且更准确（MS COCO AP50 ... 95和AP50）。
本文已经验证了大量的特征，并选择使用这些特征来提高分类和检测的精度。
这些特性可以作为未来研究和开发的最佳实践。

[论文](https://arxiv.org/pdf/2004.10934.pdf)：
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [模型架构](#目录)

选择CSPDarknet53主干、SPP附加模块、PANet路径聚合网络和YOLOv4（基于锚点）头作为YOLOv4架构。

# [预训练模型](#目录)

YOLOv4需要CSPDarknet53主干来提取图像特征进行检测。 您可以从我们的ModelZoo中获取CSPDarknet53训练脚本，并根据```./src.cspdarknet53```中的CSPDarknet53修改主干结构，最后在imagenet2012上训练它，以获得CSPDarknet53预训练模型。
步骤：

1. 从ModelZoo获取resnet50训练脚本。
2. 根据```./src.cspdarknet53``中的CSPDarknet53修改网络架构。
3. 在imagenet2012上训练CSPDarknet53。

# [数据集](#目录)

使用的数据集：[COCO2017](https://cocodataset.org/#download)  
支持的数据集：[COCO2017]或与MS COCO格式相同的数据集  
支持的标注：[COCO2017]或与MS COCO相同格式的标注

- 目录结构如下，由用户定义目录和文件的名称：

    ```text
        ├── dataset
            ├── YOLOv4
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─train
                │   ├─picture1.jpg
                │   ├─ ...
                │   └─picturen.jpg
                ├─ val
                    ├─picture1.jpg
                    ├─ ...
                    └─picturen.jpg
    ```

建议用户使用MS COCO数据集来体验模型，
其他数据集需要使用与MS COCO相同的格式。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [快速入门](#目录)

- 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：
- 在运行网络之前，准备CSPDarknet53.ckpt和hccl_8p.json文件。
    - 请参考[预训练模型]。

    - 生成hccl_8p.json，运行model_zoo/utils/hccl_tools/hccl_tools.py脚本。  
      以下参数“[0-8)”表示生成0~7卡的hccl_8p.json文件。

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- 本地运行

  ```text
  # training_shape参数用于定义网络图像形状，默认为
                     [416, 416],
                     [448, 448],
                     [480, 480],
                     [512, 512],
                     [544, 544],
                     [576, 576],
                     [608, 608],
                     [640, 640],
                     [672, 672],
                     [704, 704],
                     [736, 736].
  # 意思是使用11种形状作为输入形状，或者可以设置某种形状。

  # 使用python命令执行单尺度训练示例（1卡）
  python train.py \
      --data_dir=./dataset/xxx \
      --pretrained_backbone=cspdarknet53_backbone.ckpt \
      --is_distributed=0 \
      --lr=0.1 \
      --t_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # 使用shell脚本执行单尺度单机训练示例（1卡）
  bash run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt

  # 在Ascend设备上，使用shell脚本执行多尺度分布式训练示例（8卡）
  bash run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json

  # 使用python命令评估
  python eval.py \
      --data_dir=./dataset/xxx \
      --pretrained=yolov4.ckpt \
      --testing_shape=608 > log.txt 2>&1 &

  # 使用shell脚本评估
  bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt
  ```

- [ModelArts](https://support.huaweicloud.com/modelarts/)上训练

  ```python
  # 在Ascend上训练8卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
  #          在base_config.yaml文件中设置“pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_pretrain/”。
  #          在网站UI界面上添加“pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制预训练的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“train.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上训练1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
  #          在base_config.yaml文件中设置“pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“warmup_epochs=4”。
  #          在base_config.yaml文件中设置“training_shape=416”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_pretrain/”。
  #          在网站UI界面上添加“pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“warmup_epochs=4”。
  #          在网站UI界面添加“training_shape=416”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制预训练的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“train.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上评估1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置“pretrained='/cache/checkpoint_path/model.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“"per_batch_size=1”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_trained_ckpt/”。
  #          在网站UI界面上添加“pretrained=/cache/checkpoint_path/model.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“per_batch_size=1”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“eval.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上测试1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置“pretrained='/cache/checkpoint_path/model.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“"per_batch_size=1”。
  #          在base_config.yaml文件中设置“test_nms_thresh=0.45”。
  #          在base_config.yaml文件中设置“test_ignore_threshold=0.001”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_trained_ckpt/”。
  #          在网站UI界面上添加“pretrained=/cache/checkpoint_path/model.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“per_batch_size=1”。
  #          在网站UI界面添加“test_nms_thresh=0.45”。
  #          在网站UI界面添加“test_ignore_threshold=0.001”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“test.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  ```

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

```text
└─yolov4
  ├─README.md
  ├─README_CN.md
  ├─mindspore_hub_conf.py             # Mindspore Hub配置
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练（1卡）
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练（8卡）
    └─run_eval.sh                     # 在Ascend中启动评估
    ├─run_test.sh                     # 在Ascend中启动测试
  ├─src
    ├─__init__.py                     # Python初始化文件
    ├─config.py                       # 参数配置
    ├─cspdarknet53.py                 # 网络主干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─export.py                       # 将MindSpore模型转换为AIR模型
    ├─initializer.py                  # 参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # YOLOv4网络
    ├─yolo_dataset.py                 # 为YOLOv4创建数据集
  ├─eval.py                           # 评估验证结果
  ├─test.py#                          # 评估测试结果
  └─train.py                          # 训练网络
```

## [脚本参数](#目录)

train.py中主要参数如下：

```text
可选参数：
  -h, --help            显示此帮助消息并退出
  --device_target       实现代码的设备：“Ascend”为默认值
  --data_dir DATA_DIR   训练数据集目录
  --per_batch_size PER_BATCH_SIZE
                        训练的批处理大小。 默认值：8。
  --pretrained_backbone PRETRAINED_BACKBONE
                        CspDarkNet53的ckpt文件。 默认值：""。
  --resume_yolov4 RESUME_YOLOV4
                        YOLOv4的ckpt文件，用于微调。
                        默认值：""
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，取值选项：exponential，
                        cosine_annealing。 默认值：exponential
  --lr LR               学习率。 默认值：0.001
  --lr_epochs LR_EPOCHS
                        LR变化轮次，用“,”分隔。
                        默认值：220,250
  --lr_gamma LR_GAMMA   将LR降低一个exponential lr_scheduler因子。
                        默认值：0.1
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。 默认值：0
  --T_max T_MAX         cosine_annealing调度器中的T-max。 默认值：320
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。 默认值：320
  --warmup_epochs WARMUP_EPOCHS
                        热身轮次。 默认值：0
  --weight_decay WEIGHT_DECAY
                        权重衰减因子。 默认值：0.0005
  --momentum MOMENTUM   动量。 默认值：0.9
  --loss_scale LOSS_SCALE
                        静态损失尺度。 默认值：64
  --label_smooth LABEL_SMOOTH
                        CE中是否使用标签平滑。 默认值：0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        原one-hot的光滑强度。 默认值：0.1
  --log_interval LOG_INTERVAL
                        日志记录间隔步数。 默认值：100
  --ckpt_path CKPT_PATH
                        Checkpoint保存位置。 默认值：outputs/
  --ckpt_interval CKPT_INTERVAL
                        保存checkpoint间隔。 默认值：None
  --is_save_on_master IS_SAVE_ON_MASTER
                        在master或all rank上保存ckpt，1代表master，0代表
                        all ranks。 默认值：1
  --is_distributed IS_DISTRIBUTED
                        是否分发训练，1代表是，0代表否。 默认值：
                        1
  --rank RANK           分布式本地进程序号。 默认值：0
  --group_size GROUP_SIZE
                        设备进程总数。 默认值：1
  --need_profiler NEED_PROFILER
                        是否使用profiler。 0表示否，1表示是。 默认值：0
  --training_shape TRAINING_SHAPE
                        恢复训练形状。 默认值：""
  --resize_rate RESIZE_RATE
                        多尺度训练的缩放速率。 默认值：10
```

## [训练过程](#目录)

可以从头开始训练YLOv4，也可以使用cspdarknet53主干训练。
Cspdarknet53是一个分类器，可以在ImageNet(ILSVRC2012)等数据集上训练。
用户可轻松训练Cspdarknet53。 只需将分类器Resnet50的主干替换为cspdarknet53。
可在MindSpore ModelZoo中轻松获取Resnet50。

### 训练

在Ascend设备上，使用shell脚本执行单机训练示例（1卡）

```bash
bash run_standalone_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt
```

```text
python train.py \
    --data_dir=/dataset/xxx \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

上述python命令将在后台运行，您可以通过log.txt文件查看结果。

训练结束后，您可在默认输出文件夹下找到checkpoint文件。 得到如下损失值：

```text

# grep "loss:" train/log.txt
2020-10-16 15:00:37,483:INFO:epoch[0], iter[0], loss:8248.610352, 0.03 imgs/sec, lr:2.0466639227834094e-07
2020-10-16 15:00:52,897:INFO:epoch[0], iter[100], loss:5058.681709, 51.91 imgs/sec, lr:2.067130662908312e-05
2020-10-16 15:01:08,286:INFO:epoch[0], iter[200], loss:1583.772806, 51.99 imgs/sec, lr:4.1137944208458066e-05
2020-10-16 15:01:23,457:INFO:epoch[0], iter[300], loss:1229.840823, 52.75 imgs/sec, lr:6.160458724480122e-05
2020-10-16 15:01:39,046:INFO:epoch[0], iter[400], loss:1155.170310, 51.32 imgs/sec, lr:8.207122300518677e-05
2020-10-16 15:01:54,138:INFO:epoch[0], iter[500], loss:920.922433, 53.02 imgs/sec, lr:0.00010253786604152992
2020-10-16 15:02:09,209:INFO:epoch[0], iter[600], loss:808.610681, 53.09 imgs/sec, lr:0.00012300450180191547
2020-10-16 15:02:24,240:INFO:epoch[0], iter[700], loss:621.931513, 53.23 imgs/sec, lr:0.00014347114483825862
2020-10-16 15:02:39,280:INFO:epoch[0], iter[800], loss:527.155985, 53.20 imgs/sec, lr:0.00016393778787460178
...
```

### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例（8卡）

```bash
bash run_distribute_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt rank_table_8p.json
```

上述shell脚本将在后台运行分布式训练。 您可以通过train_parallel[X]/log.txt文件查看结果。 得到如下损失值：

```text
# 分布式训练结果(8卡，动态形状)
...
2020-10-16 20:40:17,148:INFO:epoch[0], iter[800], loss:283.765033, 248.93 imgs/sec, lr:0.00026233625249005854
2020-10-16 20:40:43,576:INFO:epoch[0], iter[900], loss:257.549973, 242.18 imgs/sec, lr:0.00029508734587579966
2020-10-16 20:41:12,743:INFO:epoch[0], iter[1000], loss:252.426355, 219.43 imgs/sec, lr:0.00032783843926154077
2020-10-16 20:41:43,153:INFO:epoch[0], iter[1100], loss:232.104760, 210.46 imgs/sec, lr:0.0003605895326472819
2020-10-16 20:42:12,583:INFO:epoch[0], iter[1200], loss:236.973975, 217.47 imgs/sec, lr:0.00039334059692919254
2020-10-16 20:42:39,004:INFO:epoch[0], iter[1300], loss:228.881298, 242.24 imgs/sec, lr:0.00042609169031493366
2020-10-16 20:43:07,811:INFO:epoch[0], iter[1400], loss:255.025714, 222.19 imgs/sec, lr:0.00045884278370067477
2020-10-16 20:43:38,177:INFO:epoch[0], iter[1500], loss:223.847151, 210.76 imgs/sec, lr:0.0004915939061902463
2020-10-16 20:44:07,766:INFO:epoch[0], iter[1600], loss:222.302487, 216.30 imgs/sec, lr:0.000524344970472157
2020-10-16 20:44:37,411:INFO:epoch[0], iter[1700], loss:211.063779, 215.89 imgs/sec, lr:0.0005570960929617286
2020-10-16 20:45:03,092:INFO:epoch[0], iter[1800], loss:210.425542, 249.21 imgs/sec, lr:0.0005898471572436392
2020-10-16 20:45:32,767:INFO:epoch[1], iter[1900], loss:208.449521, 215.67 imgs/sec, lr:0.0006225982797332108
2020-10-16 20:45:59,163:INFO:epoch[1], iter[2000], loss:209.700071, 242.48 imgs/sec, lr:0.0006553493440151215
...
```

### 迁移学习

可以基于预训练分类或检测模型来训练自己的模型。 按照以下步骤进行迁移学习。

1. 将数据集转换为COCO样式。 否则，必须添加自己的数据预处理代码。
2. 根据自己的数据集设置config.py，特别是`num_classes`。
3. 在调用`train.py`时，将参数`filter_weight`设置为`True`，`pretrained_checkpoint`设置为预训练检查点，这样将从预训练模型中筛选出最终的检测框权重。
4. 使用新的配置和参数构建自己的bash脚本。

## [评估过程](#目录)

### 验证

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

上述python命令将在后台运行。 您可以通过log.txt文件查看结果。 测试数据集的mAP如下：

```text
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
```

### Test-dev

```bash
python test.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_test.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

predict_xxx.json文件位于test/outputs/%Y-%m-%d_time_%H_%M_%S/。
将文件predict_xxx.json重命名为detections_test-dev2017_yolov4_results.json，并将其压缩为detections_test-dev2017_yolov4_results.zip。
将detections_test-dev2017_yolov4_results.zip文件提交到MS COCO评估服务器用于test-dev 2019 (bbox) <https://competitions.codalab.org/competitions/20794#participate>。
您将在文件末尾获得这样的结果。查看评分输出日志。

```text
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.642
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
```

## [转换过程](#目录)

### 转换

如果您想推断Ascend 310上的网络，则应将模型转换为MINDIR：

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`EXPORT_FORMAT`取值为["AIR", "ONNX", "MINDIR"]。

## [推理过程](#目录)

### 用法

在执行推理之前，必须在910环境上通过导出脚本导出MINDIR文件。
当前批处理大小只能设置为1。 精度计算过程需要70G+内存空间。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID`是可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在ac.log文件中找到类似如下结果。

```text
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.438
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.630
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.475
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.330
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.542
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.588
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
```

# [模型说明](#目录)

## [性能](#目录)

### 评估性能

YOLOv4应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

|参数| YOLOv4 |
| -------------------------- | ----------------------------------------------------------- |
|资源| Ascend 910；CPU 2.60GHz, 192核；内存：755G；系统：EulerOS 2.8；|
|上传日期|2020年10月16日|
| MindSpore版本|1.0.0-alpha|
|数据集|118000张图像|
|训练参数|epoch=320, batch_size=8, lr=0.012,momentum=0.9|
| 优化器                  | Momentum                                                    |
|损失函数|Sigmoid Cross Entropy with logits, Giou Loss|
|输出|框和标签|
|损失| 50 |
|速度| 1卡：53FPS；8卡：390FPS (shape=416) 220FPS (动态形状)|
|总时长|48小时（动态形状）|
|微调检查点|约500M（.ckpt文件）|
|脚本| <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/> |

### 推理性能

YOLOv4应用于20000张图像上（标注和数据格式必须与COCO test 2017相同）

|参数| YOLOv4 |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G             |
|上传日期|2020年10月16日|
| MindSpore版本|1.0.0-alpha|
|数据集|20000张图像|
|批处理大小|1|
|输出|边框位置和分数，以及概率|
|精度|map >= 44.7%(shape=608)|
|推理模型|约500M（.ckpt文件）|

# [随机情况说明](#目录)

在dataset.py中，我们设置了“create_dataset”函数内的种子。
在var_init.py中，我们设置了权重初始化的种子。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
