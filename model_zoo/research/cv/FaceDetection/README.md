# Contents

- [Face Detection Description](#face-detection-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)

# [Face Detection Description](#contents)

This is a Face Detection network based on Yolov3, with support for training and evaluation on Ascend910.

You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 is extremely fast and accurate.

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
 YOLOv3 use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi,
University of Washington

# [Model Architecture](#contents)

Face Detection uses a modified-DarkNet53 network for performing feature extraction. It has 45 convolutional layers.

# [Dataset](#contents)

We use about 13K images as training dataset and 3K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. WiderFace)

- step 1: The dataset should follow the Pascal VOC data format for object detection. The directory structure is as follows:(Because of the small input shape of network, we remove the face lower than 50*50 at 1080P in evaluating dataset )

    ```python
        .
        └─ dataset
          ├─ Annotations
            ├─ img1.xml
            ├─ img2.xml
            ├─ ...
          ├─ JPEGImages
            ├─ img1.jpg
            ├─ img2.jpg
            ├─ ...
          └─ ImageSets
            └─ Main
              └─ train.txt or test.txt
    ```

- step 2: Convert the dataset to mindrecord:

    ```bash
    python data_to_mindrecord_train.py
    ```

    or

    ```bash
    python data_to_mindrecord_eval.py
    ```

    If your dataset is too big to convert at a time, you can add data to an existed mindrecord in turn:

    ```shell
    python data_to_mindrecord_train_append.py
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend, CPU, GPU）
    - Prepare hardware environment with Ascend, CPU or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```text
.
└─ Face Detection
  ├─ README.md
  ├─ ascend310_infer                        # application for 310 inference
  ├─ model_utils
    ├─ __init__.py                          # init file
    ├─ config.py                            # Parse arguments
    ├─ device_adapter.py                    # Device adapter for ModelArts
    ├─ local_adapter.py                     # Local adapter
    └─ moxing_adapter.py                    # Moxing adapter for ModelArts
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_standalone_train_gpu.sh          # launch standalone training(1p) in GPU
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_distribute_train_gpu.sh          # launch distributed training(8p) in GPU
    ├─ run_eval.sh                          # launch evaluating in ascend
    ├─ run_infer_310.sh                     # launch inference on Ascend310
    └─ run_export.sh                        # launch exporting air model
  ├─ src
    ├─ FaceDetection
      ├─ voc_wrapper.py                     # get detection results
      ├─ yolo_loss.py                       # loss function
      ├─ yolo_postprocess.py                # post process
      └─ yolov3.py                          # network
    ├─ data_preprocess.py                   # preprocess
    ├─ logging.py                           # log function
    ├─ lrsche_factory.py                    # generate learning rate
    ├─ network_define.py                    # network define
    ├─ transforms.py                        # data transforms
    ├─ data_to_mindrecord_train.py          # convert dataset to mindrecord for training
    ├─ data_to_mindrecord_train_append.py   # add dataset to an existed mindrecord for training
    └─ data_to_mindrecord_eval.py           # convert dataset to mindrecord for evaluating
  ├─ default_config.yaml                    # default configurations
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  ├─ postprocess.py                         # postprocess script
  ├─ preprocess.py                          # preprocess script
  ├─ bin.py                                 # bin script
  └─ export.py                              # export air model
```

## [Running Example](#contents)

### Train

- Stand alone mode

    # Training on Ascend

    ```bash
    cd ./scripts
    bash run_standalone_train.sh [PLATFORM] [MINDRECORD_FILE] [USE_DEVICE_ID]
    ```

    or (fine-tune)

    ```bash
    cd ./scripts
    bash run_standalone_train.sh [PLATFORM] [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    bash run_standalone_train.sh CPU /home/train.mindrecord 0 /home/a.ckpt
    ```

    # Training on GPU

    ```python
    python ./train.py --config_path=./default_config.yaml
    ```

    ```bash
    cd ./scripts
    bash run_standalone_train_gpu.sh [CONFIG_PATH]
    ```

- Distribute mode (recommended)

    # Distribute training on Ascend

    ```bash
    cd ./scripts
    bash run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE]
    ```

    or (fine-tune)

    ```bash
    cd ./scripts
    bash run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

    for example:

    ```bash
    cd ./scripts
    bash run_distribute_train.sh /home/train.mindrecord ./rank_table_8p.json /home/a.ckpt
    ```

    # Distribute training on GPU

    ```python
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
        python ./train.py \
        --config_path=$CONFIG_PATH > train.log  2>&1 &
    ```

    ```bash
    cd ./scripts
    bash run_distribute_train_gpu.sh [CONFIG_PATH]
    ```

    *Distribute mode doesn't support running on CPU*. You will get the loss value of each step as following in "./scripts/device0/output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

    ```python
    rank[0], iter[0], loss[318555.8], overflow:False, loss_scale:1024.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[1], loss[95394.28], overflow:True, loss_scale:1024.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[2], loss[81332.92], overflow:True, loss_scale:512.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[3], loss[27250.805], overflow:True, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    ...
    rank[0], iter[62496], loss[2218.6282], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[62497], loss[3788.5146], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[62498], loss[3427.5479], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    rank[0], iter[62499], loss[4294.194], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
    ```

- Train on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```text
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_train/data.mindrecord'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
    #          (optional)Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_train/data.mindrecord'" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrain/'" on the website UI interface.
    #          (optional)Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) (optional) Upload or copy your pretrained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceDetection" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "run_platform='Ascend'" on default_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_train/data.mindrecord'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
    #          (optional)Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "run_platform='Ascend'" on the website UI interface.
    #          Add "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_train/data.mindrecord'" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrain/'" on the website UI interface.
    #          (optional)Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) (optional) Upload or copy your pretrained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceDetection" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "run_platform='Ascend'" on default_config.yaml file.
    #          Set "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_train/data.mindrecord'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "run_platform='Ascend'" on the website UI interface.
    #          Add "mindrecord_path='/cache/data/face_detect_dataset/mindrecord_test/data.mindrecord'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_pretrain/'" on the website UI interface.
    #          Add "pretrained='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your pretrained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceDetection" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Export 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on default_config.yaml file.
    #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set "batch_size=1" on default_config.yaml file.
    #          Set "file_format='AIR'" on default_config.yaml file.
    #          Set "file_name='FaceDetection'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
    #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add "batch_size=1" on the website UI interface.
    #          Add "file_format=AIR" on the website UI interface.
    #          Add "file_name=FaceDetection" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/FaceDetection" on the website UI interface.
    # (6) Set the startup file to "export.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

### Evaluation

```bash
cd ./scripts
bash run_eval.sh [PLATFORM] [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

```python
GPU platform
python eval.py [CONFIG_PATH]
```

for example:

```bash
cd ./scripts
bash run_eval.sh Ascend /home/eval.mindrecord 0 /home/a.ckpt
```

```python
GPU platform
python eval.py --config_path=./default_config.yaml
```

You will get the result as following in "./scripts/device0/eval.log":

```python
calculate [recall | persicion | ap]...
Saving ../../results/0-2441_61000/.._.._results_0-2441_61000_face_AP_0.760.png
```

And the detect result and P-R graph will also be saved in "./results/[MODEL_NAME]/"

### Inference process

#### Convert model

If you want to infer the network on Ascend 310, you should convert the model to MINDIR or AIR:

```shell
# Ascend310 inference
python export.py --pretrained [PRETRAIN] --batch_size [BATCH_SIZE] --file_format [EXPORT_FORMAT]
```

The pretrained parameter is required.
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
Current batch_size can only be set to 1.

#### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [MINDRECORD_PATH] [DEVICE_ID]
```

- `DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in current path, you can find result like this in map.log file.

```bash
calculate [recall | persicion | ap]...
Saving ../../results/0-2441_61000/.._.._results_0-2441_61000_face_AP_0.7575.png
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Face Detection                                              | Face Detection                                              |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | GPU PCIE V100-32G                                           |
| uploaded Date              | 09/30/2020 (month/day/year)                                 | 07/01/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       | 1.3.0                                                       |
| Dataset                    | 13K images                                                  | 13K images                                                  |
| Training Parameters        | epoch=2500, batch_size=64, momentum=0.5                     | epoch=2500, batch_size=64, momentum=0.5                     |
| Optimizer                  | Momentum                                                    | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| outputs                    | boxes and label                                             | boxes and label                                             |
| Speed                      | 1pc: 800~850 ms/step; 8pcs: 1000~1150 ms/step               | 1pc: 180fps; 4pcs: 385fps                                   |
| Total time                 | 1pc: 120 hours; 8pcs: 18 hours                              | 4pcs: 23 hours                                              |
| Checkpoint for Fine tuning | 37M (.ckpt file)                                            | --                                                          |

### Evaluation Performance

| Parameters          | Face Detection              | Face Detection              |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | V1                          | V1                          |
| Resource            | Ascend 910; OS Euler2.8     | GPU NV SMX2 V100-32G        |
| Uploaded Date       | 09/30/2020 (month/day/year) | 07/01/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.3.0                       |
| Dataset             | 3K images                   | 3K images                   |
| batch_size          | 1                           | 1                           |
| outputs             | mAP                         | mAP                         |
| Accuracy            | 8pcs: 76.0%                 | 4pcs: 77.8%                 |
| Model for inference | 37M (.ckpt file)            | --                          |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Face Detection              |
| Resource            | Ascend 310; Euler2.8        |
| Uploaded Date       | 19/06/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | 3K images                   |
| batch_size          | 1                           |
| outputs             | mAP                         |
| mAP                 | mAP=75.75%                  |
| Model for inference | 37M(.ckpt file)             |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
