# Contents

- [Contents](#contents)
- [YOLOv3_ResNet18 Description](#yolov3_resnet18-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training on Ascend](#training-on-ascend)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation on Ascend](#evaluation-on-ascend)
    - [Export MindIR](#export-mindir)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv3_ResNet18 Description](#contents)

YOLOv3 network based on ResNet-18, with support for training and evaluation.

[Paper](https://arxiv.org/abs/1804.02767):  Joseph Redmon, Ali Farhadi. arXiv preprint arXiv:1804.02767, 2018. 2, 4, 7, 11.

# [Model Architecture](#contents)

The overall network architecture of YOLOv3 is show below:

And we use ResNet18 as the backbone of YOLOv3_ResNet18. The architecture of ResNet18 has 4 stages. The ResNet architecture performs the initial convolution and max-pooling using 7×7 and 3×3 kernel sizes respectively. Afterward,  every stage of the network has different Residual blocks (2, 2, 2, 2) containing two 3×3 conv layers. Finally, the network has an Average Pooling layer followed by a fully connected layer.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

- Dataset

    1. The directory structure is as follows:

        ```
        .
        ├── annotations  # annotation jsons
        ├── train2017    # train dataset
        └── val2017      # infer dataset
        ```

    2. Organize the dataset information into a TXT file, each row in the file is as follows:

        ```
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. `dataset.py` is the parsing script, we read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are external inputs.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation on Ascend as follows:

- Running on Ascend

    ```shell script
    #run standalone training example
    bash run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]

    #run distributed training example
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] [RANK_TABLE_FILE]

    #run evaluation example
    bash run_eval.sh [DEVICE_ID] [CKPT_PATH] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH]
    ```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='coco'" on default_config.yaml file.
    #          Set "lr=0.005" on default_config.yaml file.
    #          Set "mindrecord_dir='/cache/data/coco/Mindrecord_train'" on default_config.yaml file.
    #          Set "image_dir='/cache/data'" on default_config.yaml file.
    #          Set "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'" on default_config.yaml file.
    #          Set "epoch_size=160" on default_config.yaml file.
    #          (optional)Set "pre_trained_epoch_size=YOUR_SIZE" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          (optional)Set "pre_trained=/cache/checkpoint_path/model.ckpt" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='coco'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "lr=0.005" on the website UI interface.
    #          Add "mindrecord_dir=/cache/data/coco/Mindrecord_train" on the website UI interface.
    #          Add "image_dir=/cache/data" on the website UI interface.
    #          Add "anno_path=/cache/data/coco/train_Person+Face-coco-20190118.txt" on the website UI interface.
    #          Add "epoch_size=160" on the website UI interface.
    #          (optional)Add "pre_trained_epoch_size=YOUR_SIZE" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          (optional)Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "train.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR --anno_path=$ANNO_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/yolov3_resnet18" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='coco'" on default_config.yaml file.
    #          Set "mindrecord_dir='/cache/data/coco/Mindrecord_train'" on default_config.yaml file.
    #          Set "image_dir='/cache/data'" on default_config.yaml file.
    #          Set "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'" on default_config.yaml file.
    #          Set "epoch_size=160" on default_config.yaml file.
    #          (optional)Set "pre_trained_epoch_size=YOUR_SIZE" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          (optional)Set "pre_trained=/cache/checkpoint_path/model.ckpt" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='coco'" on the website UI interface.
    #          Add "mindrecord_dir='/cache/data/coco/Mindrecord_train'" on the website UI interface.
    #          Add "image_dir='/cache/data'" on the website UI interface.
    #          Add "anno_path='/cache/data/coco/train_Person+Face-coco-20190118.txt'" on the website UI interface.
    #          Add "epoch_size=160" on the website UI interface.
    #          (optional)Add "pre_trained_epoch_size=YOUR_SIZE" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          (optional)Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "train.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR --anno_path=$ANNO_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/yolov3_resnet18" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='coco'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "ckpt_path='/cache/checkpoint_path/yolov3-160_156.ckpt'" on default_config.yaml file.
    #          Set "eval_mindrecord_dir='/cache/data/coco/Mindrecord_eval'" on default_config.yaml file.
    #          Set "image_dir='/cache/data'" on default_config.yaml file.
    #          Set "anno_path='/cache/data/coco/test_Person+Face-coco-20190118.txt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='coco'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "ckpt_path='/cache/checkpoint_path/yolov3-160_156.ckpt'" on the website UI interface.
    #          Add "eval_mindrecord_dir='/cache/data/coco/Mindrecord_eval'" on the website UI interface.
    #          Add "image_dir='/cache/data'" on the website UI interface.
    #          Add "anno_path='/cache/data/coco/test_Person+Face-coco-20190118.txt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "eval.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python eval.py --only_create_dataset=True --eval_mindrecord_dir=$EVAL_MINDRECORD_DIR --image_dir=$EVAL_IMAGE_DIR --anno_path=$EVAL_ANNO_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/yolov3_resnet18" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└── cv
    ├── README.md                           // descriptions about all the models
    ├── mindspore_hub_conf.md               // config for mindspore hub
    └── yolov3_resnet18
        ├── README.md                       // descriptions about yolov3_resnet18
        ├── README_CN.md                    // descriptions about yolov3_resnet18 with Chinese
        ├── model_utils
            ├── __init__.py                 // init file
            ├── config.py                   // Parse arguments
            ├── device_adapter.py           // Device adapter for ModelArts
            ├── local_adapter.py            // Local adapter
            └── moxing_adapter.py           // Moxing adapter for ModelArts
        ├── scripts
            ├── run_distribute_train.sh     // shell script for distributed on Ascend
            ├── run_standalone_train.sh     // shell script for distributed on Ascend
            └── run_eval.sh                 // shell script for evaluation on Ascend
        ├── src
            ├── dataset.py                  // creating dataset
            ├── yolov3.py                   // yolov3 architecture
            ├── config.py                   // default arguments for network architecture
            └── utils.py                    // util function
        ├── default_config.yaml             // configurations
        ├── eval.py                         // evaluation script
        ├── export.py                       // export script
        ├── mindspore_hub_conf.py           // hub config
        ├── postprocess.py                  // postprocess script
        └── train.py                        // train script
```

## [Script Parameters](#contents)

  Major parameters in train.py and config.py as follows:

  ```python
    device_num: Use device nums, default is 1.
    lr: Learning rate, default is 0.001.
    epoch_size: Epoch size, default is 50.
    batch_size: Batch size, default is 32.
    pre_trained: Pretrained Checkpoint file path.
    pre_trained_epoch_size: Pretrained epoch size.
    mindrecord_dir: Mindrecord directory.
    image_dir: Dataset path.
    anno_path: Annotation path.

    img_shape: Image height and width used as input to the model.
  ```

## [Training Process](#contents)

### Training on Ascend

To train the model, run `train.py` with the dataset `image_dir`, `anno_path` and `mindrecord_dir`. If the `mindrecord_dir` is empty, it wil generate [mindrecord](https://www.mindspore.cn/docs/programming_guide/en/master/convert_dataset.html) file by `image_dir` and `anno_path`(the absolute image path is joined by the `image_dir` and the relative path in `anno_path`). **Note if `mindrecord_dir` isn't empty, it will use `mindrecord_dir` rather than `image_dir` and `anno_path`.**

- Stand alone mode

    ```bash
    bash run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt
    ```

    The input variables are device id, epoch size, mindrecord directory path, dataset directory path and train TXT file path.

- Distributed mode

    ```bash
    bash run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json
    ```

    The input variables are device numbers, epoch size, mindrecord directory path, dataset directory path, train TXT file path and [hccl json configuration file](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools). **It is better to use absolute path.**

You will get the loss value and time of each step as following:

  ```bash
  epoch: 145 step: 156, loss is 12.202981
  epoch time: 25599.22742843628, per step time: 164.0976117207454
  epoch: 146 step: 156, loss is 16.91706
  epoch time: 23199.971675872803, per step time: 148.7177671530308
  epoch: 147 step: 156, loss is 13.04007
  epoch time: 23801.95164680481, per step time: 152.57661312054364
  epoch: 148 step: 156, loss is 10.431475
  epoch time: 23634.241580963135, per step time: 151.50154859591754
  epoch: 149 step: 156, loss is 14.665991
  epoch time: 24118.8325881958, per step time: 154.60790120638333
  epoch: 150 step: 156, loss is 10.779521
  epoch time: 25319.57221031189, per step time: 162.30495006610187
  ```

Note the results is two-classification(person and face) used our own annotations with coco2017, you can change `num_classes` in `config.py` to train your dataset. And we will support 80 classifications in coco2017 the near future.

## [Evaluation Process](#contents)

### Evaluation on Ascend

To eval, run `eval.py` with the dataset `image_dir`, `anno_path`(eval txt), `mindrecord_dir` and `ckpt_path`. `ckpt_path` is the path of [checkpoint](https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html) file.

  ```bash
  bash run_eval.sh 0 yolo.ckpt ./Mindrecord_eval ./dataset ./dataset/eval.txt
  ```

The input variables are device id, checkpoint path, mindrecord directory path, dataset directory path and train TXT file path.

You will get the precision and recall value of each class:

  ```bash
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

Note the precision and recall values are results of two-classification(person and face) used our own annotations with coco2017.

## [Export MindIR](#contents)

Currently, batchsize can only set to 1.

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Inference Process](#contents)

### Usage

Before performing inference, the mindir file must be exported by export.py.
Current batch_Size can only be set to 1. Images to be processed needs to be copied to the to-be-processed folder based on the annotation file.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

  ```bash
  class 0 precision is 88.18%, recall is 66.00%
  class 1 precision is 85.34%, recall is 79.13%
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | YOLOv3_Resnet18 V1                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 07/05/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | COCO2017                                                    |
| Training Parameters        | epoch = 160, batch_size = 32, lr = 0.005                    |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy                                       |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 120 ms/step;  8pcs: 160 ms/step                        |
| Total time                 | 1pc: 150 mins;  8pcs: 70 mins                               |
| Parameters (M)             | 189                                                         |
| Scripts                    | [yolov3_resnet18 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) | [yolov3_resnet18 script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18) |

### Inference Performance

| Parameters          | Ascend                                          |
| ------------------- | ----------------------------------------------- |
| Model Version       | YOLOv3_Resnet18 V1                              |
| Resource            | Ascend 910; OS Euler2.8                         |
| Uploaded Date       | 07/05/2021 (month/day/year)                     |
| MindSpore Version   | 1.3.0                                           |
| Dataset             | COCO2017                                        |
| batch_size          | 1                                               |
| outputs             | presion and recall                              |
| Accuracy            | class 0: 88.18%/66.00%; class 1: 85.34%/79.13%  |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  

