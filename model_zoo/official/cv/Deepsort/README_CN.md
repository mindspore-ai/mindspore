# 目录

<!-- TOC -->

- [目录](#目录)
- [DeepSort描述](#DeepSort描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## DeepSort描述

DeepSort是2017年提出的多目标跟踪算方法。该网络在MOT16获得冠军，不仅提升了精度，而且速度比之前快20倍。

[论文](https://arxiv.org/abs/1602.00763)： Nicolai Wojke, Alex Bewley, Dietrich Paulus. "SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC". *Presented at ICIP 2016*.

## 模型架构

DeepSort由一个特征提取器、一个卡尔曼滤波和一个匈牙利算法组成。特征提取器用于提取框中人物特征信息，卡尔曼滤波根据上一帧信息预测当前帧人物位置，匈牙利算法用于匹配预测信息与检测到的人物位置信息。

## 数据集

使用的数据集：[MOT16](<https://motchallenge.net/data/MOT16.zip>)、[Market-1501](<https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view>)

MOT16:

- 数据集大小：1.9G，共14个视频帧序列
    - test：7个视频序列帧
    - train：7个序列帧
- 数据格式(一个train视频帧序列)：
    - det:视频序列中人物坐标以及置信度等信息
    - gt:视频跟踪标签信息
    - img1:视频中所有帧序列
    - 注意：由于作者提供的视频帧序列检测到的坐标信息和置信度信息不一样，所以在跟踪时使用作者提供的信息，作者提供的[npy](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)文件。

Market-1501:

- 使用：
    - 使用目的：训练DeepSort特征提取器
    - 使用方法： 先使用prepare.py处理数据

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，提取det信息(使用作者提供的检测框信息)，在脚本中给出数据路径
python process-npy.py
# 进入脚本目录，预处理数据集(Market-1501)，在脚本中给出数据集路径
python prepare.py
# 进入脚本目录，训练DeepSort特征提取器
python src/deep/train.py --run_modelarts=False --run_distribute=True --data_url="" --train_url=""
# 进入脚本目录，提取detections信息
python generater_detection.py --run_modelarts=False --run_distribute=True --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
# 进入脚本目录，生成跟踪信息
python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""

#Ascend多卡训练
bash scripts/run_distribute_train.sh train_code_path  RANK_TABLE_FILE DATA_PATH
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```bash
├── DeepSort
    ├── scripts
    │   ├──run_distribute_train.sh          // 在Ascend中多卡训练
    ├── src             //源码
    │   │   ├── application_util
    │   │   │   ├──image_viewer.py
    │   │   │   ├──preprocessing.py
    │   │   │   ├──visualization.py
    │   │   ├──deep
    │   │   │   ├──feature_extractor.py     //提取目标框中人物特征信息
    │   │   │   ├──original_model.py       //特征提取器模型
    │   │   │   ├──train.py                //训练网络模型
    │   │   ├──sort
    │   │   │   ├──detection.py
    │   │   │   ├──iou_matching.py              //预测信息与真实框匹配
    │   │   │   ├──kalman_filter.py          //卡尔曼滤波，预测跟踪框信息
    │   │   │   ├──linear_assignment.py
    │   │   │   ├──nn_matching.py         //框匹配
    │   │   │   ├──track.py             //跟踪器
    │   │   │   ├──tracker.py           //跟踪器
    ├── deep_sort_app.py                //目标跟踪
    ├── evaluate_motchallenge.py        //生成跟踪结果信息
    ├── generate_videos.py              //根据跟踪结果生成跟踪视频
    ├── generater-detection.py          //生成detection信息
    ├── postprocess.py                  //生成Ascend310推理数据
    ├── preprocess.py                 //处理Ascend310推理结果，生成精度
    ├── prepare.py                      //处理训练数据集
    ├── process-npy.py                  //提取帧序列人物坐标和置信度
    ├── show_results.py                 //展示跟踪结果
    ├── README.md                    // DeepSort相关说明
```

### 脚本参数

```python
train.py generater_detection.py evaluate_motchallenge.py 中主要参数如下:

--data_url: 到训练和提取信息数据集的绝对完整路径
--train_url: 输出文件路径。
--epoch: 总训练轮次
--batch_size: 训练批次大小
--device_targe: 实现代码的设备。值为'Ascend'
--ckpt_url: 训练后保存的检查点文件的绝对完整路径
--model_name: 模型文件名称
--det_url: 视频帧序列人物信息文件路径
--detection_url:  人物坐标信息、置信度以及特征信息文件路径
--run_distribute: 多卡运行
--run_modelarts: ModelArts上运行
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```bash
  python src/deep/train.py --run_modelarts=False --run_distribute=False --data_url="" --train_url=""
  # 或进入脚本目录，执行脚本
  bash scripts/run_distribute_train.sh train_code_path  RANK_TABLE_FILE DATA_PATH
  ```

  经过训练后，损失值如下：

  ```bash
  # grep "loss is " log
  epoch: 1 step: 3984, loss is 6.4320717
  epoch: 1 step: 3984, loss is 6.414733
  epoch: 1 step: 3984, loss is 6.4306755
  epoch: 1 step: 3984, loss is 6.4387856
  epoch: 1 step: 3984, loss is 6.463995
  ...
  epoch: 2 step: 3984, loss is 6.436552
  epoch: 2 step: 3984, loss is 6.408932
  epoch: 2 step: 3984, loss is 6.4517527
  epoch: 2 step: 3984, loss is 6.448922
  epoch: 2 step: 3984, loss is 6.4611588
  ...
  ```

  模型检查点保存在当前目录下。

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行

  ```bash
  # 进入脚本目录，提取det信息(使用作者提供的检测框信息)
    python process-npy.py
  # 进入脚本目录，提取detections信息
  python generater_detection.py --run_modelarts False --run_distribute True --data_url "" --train_url "" --det_url "" --ckpt_url "" --model_name ""
  # 进入脚本目录，生成跟踪信息
  python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
  # 生成跟踪结果
  python eval_motchallenge.py ----run_modelarts=False --data_url="" --train_url="" --result_url=""
  ```

- [测评工具](https://github.com/cheind/py-motmetrics)

  说明:脚本中引用头文件可能存在一些问题，自行修改头文件路径即可

  ```bash
  #测量精度
  python motmetrics/apps/eval_motchallenge.py --groundtruths="" --tests=""
  ```

-
  测试数据集的准确率如下：

| 数据 | MOTA | MOTP| MT | ML| IDs | FM | FP | FN |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -----------------------------------------------------------
| MOT16-02 | 29.0% | 0.207 | 11 | 11| 159 | 226 | 4151 | 8346 |
| MOT16-04 | 58.6% | 0.167| 42 | 14| 62 | 242 | 6269 | 13374 |
| MOT16-05 | 51.7% | 0.213| 31 | 27| 68 | 109 | 630 | 2595 |
| MOT16-09 | 64.3% | 0.162| 12 | 1| 39 | 58 | 309 | 1537 |
| MOT16-10 | 49.2% | 0.228| 25 | 1| 201 | 307 | 3089 | 2915 |
| MOT16-11 | 65.9% | 0.152| 29 | 9| 54 | 99 | 907 | 2162 |
| MOT16-13 | 45.0% | 0.237| 61 | 7| 269 | 335 | 3709 | 2251 |
| overall | 51.9% | 0.189| 211 | 70| 852 | 1376 | 19094 | 33190 |

## [导出mindir模型](#contents)

```shell
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH]
```

## [推理过程](#contents)

### 用法

执行推断之前，minirir文件必须由export.py导出。输入文件必须为bin格式

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

### 结果

推理结果文件保存在当前路径中，将文件作为输入，输入到eval_motchallenge.py中，然后输出result文件，输入到测评工具中即可得到精度结果。

## 模型描述

### 性能

#### 评估性能

| 参数 | ModelArts
| -------------------------- | -----------------------------------------------------------
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G
| 上传日期 | 2021-08-12
| MindSpore版本 | 1.2.0
| 数据集 | MOT16 Market-1501
| 训练参数 | epoch=100, step=191, batch_size=8, lr=0.1
| 优化器 | SGD
| 损失函数 | SoftmaxCrossEntropyWithLogits
| 损失 | 0.03
| 速度 | 9.804毫秒/步
| 总时间 | 10分钟
| 微调检查点 | 大约40M （.ckpt文件）
| 脚本 | [DeepSort脚本]

## 随机情况说明

train.py中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
