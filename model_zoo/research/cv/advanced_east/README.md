# Contents

- [Advanced East Description](#advancedeast-description)
- [Environment](#environment)
- [Dependences](#dependences)
- [Project Files](#project-files)
- [Dataset](#dataset)
- [Run The Project](#run-the-project)
    - [Data Preprocess](#data-preprocess)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Advanced East Description](#contents)

AdvancedEAST is inspired by EAST [EAST:An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The architecture of AdvancedEAST is showed below.![AdvancedEast network arch](AdvancedEast.network.png)
This project is inherited by [huoyijie/AdvancedEAST](https://github.com/huoyijie/AdvancedEAST)(preprocess, network architecture, predict) and [BaoWentz/AdvancedEAST-PyTorch](https://github.com/BaoWentz/AdvancedEAST-PyTorch)(performance).

# [Environment](#contents)

- euleros v2r7 x86_64\Ubuntu 16.04
- python 3.7.5

# [Dependences](#contents)

- mindspore==1.2.0
- shapely==1.7.1
- numpy==1.19.4
- tqdm==4.36.1

# [Project Files](#contents)

- configuration of file
    cfg.py, control parameters
- pre-process data:
    preprocess.py, resize image
- label data:
    label.py,produce label info
- define network
    model.py and VGG.py
- define loss function
    losses.py
- execute training
    advanced_east.py and dataset.py
- predict
    predict.py and nms.py
- scoring
    score.py
- logging
    logger.py

```shell
.
└──advanced_east
  ├── README.md
  ├── scripts
    ├── run_distribute_train_ascend.sh        # launch ascend distributed training(8 pcs)
    ├── run_standalone_train_ascend.sh     # launch ascend standalone training(1 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    └── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    └── eval.sh                            # evaluate model(1 pcs)
  ├── src
    ├── cfg.py                             # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── label.py                           # produce label info
    ├── logger.py                          # generate learning rate for each step
    ├── model.py                           # define network
    ├── nms.py                             # non-maximum suppression
    ├── predict.py                         # predict boxes
    ├── preprocess.py                      # pre-process data
    └── score.py                           # scoring
    └── vgg.py                             # vgg model
  ├── export.py                            # export model for inference
  ├── prepare_data.py                      # exec data preprocessing
  ├── eval.py                              # eval net
  ├── train.py                             # train net on multi-size input
  └── train_single_size.py                 # train net on fix-size input
```

# [Dataset](#contents)

ICPR MTWI 2018 challenge 2：Text detection of network image，[Link](https://tianchi.aliyun.com/competition/entrance/231651/introduction). It is not available to download dataset on the origin webpage,
the dataset is now provided by the author of the original project，[Baiduyun link](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)， password: ye9y. There are 10000 images and corresponding label
information in total in the dataset, which is divided into 2 directories with 9000 and 1000 samples respectively. In the origin training setting, training set and validation set are partitioned at the ratio
of 9:1. If you want to use your own dataset, please modify the configuration of dataset in /src/config.py. The organization of dataset file is listed as below：

  > ```bash
  > .
  > └─data_dir
  >   ├─images                # dataset
  >   └─txt                   # vertex of text boxes
  > ```
Some parameters in config.py：

```shell
    'validation_split_ratio': 0.1,      # ratio of validation dataset
    'total_img': 10000,                  # total number of samples in dataset
    'data_dir': './icpr/',              # dir of dataset
    'train_fname': 'train.txt',         # the file which stores the images file name in training dataset
    'val_fname': 'val.txt',             # the file which stores the images file name in validation dataset
    'mindsrecord_train_file': 'advanced-east.mindrecord',               # mindsrecord of training dataset
    'mindsrecord_test_file': 'advanced-east-val.mindrecord',            # mindsrecord of validation dataset
    'origin_image_dir_name': 'images_9000/',    # dir which stores the original images.
    'train_image_dir_name': 'images_train/',    # dir which stores the preprocessed images.
    'origin_txt_dir_name': 'txt_9000/',         # dir which stores the original text verteices.
    'train_label_dir_name': 'labels_train/',    # dir which stores the preprocessed text verteices.
```

# [Run The Project](#contents)

## [Data Preprocess](#contents)

Resize all the images to fixed size, and convert the label information(the vertex of text box) into the format used in training and evaluation, then the Mindsrecord files are generated.

```bash
python preparedata.py
```

## [Training Process](#contents)

Prepare the VGG16 pre-training model. Due to copyright restrictions, please go to https://github.com/machrisaa/tensorflow-vgg to download the VGG16 pre-training model and place it in the src folder.
If you have the checkpoint of VGG16, you can load the parameters in this way, the training training time can be shorten obviously.

- single Ascend

```bash
python train.py  --device_target="Ascend" --is_distributed=0 --device_id=0  > output.train.log 2>&1 &
```

- single GPU

```bash
python train.py  --device_target="GPU" --is_distributed=0 --device_id=0  > output.train.log 2>&1 &
```

- single device with specific size

```bash
python train_single_size.py  --device_target="Ascend" --is_distributed=0 --device_id=2 --size=256  > output.train.log 2>&1 &
```

- multi Ascends

```bash
# running on distributed environment（8p）
bash run_distribute_train.sh [DATSET_PATH] [RANK_TABLE_FILE]
```

The detailed training parameters are in /src/config.py。

- multi GPUs

```bash
# running on distributed environment（8p）
bash scripts/run_distribute_train_gpu.sh
```

The detailed training parameters are in /src/config.py。

config.py：

```bash
    'initial_epoch': 0, # epoch to init
    'learning_rate': 1e-4, # learning rate when initialization
    'decay': 5e-4, # weightdecay parameter
    'epsilon': 1e-4, # the value of epsilon in loss computation
    'batch_size': 8, # batch size
    'lambda_inside_score_loss': 4.0, # coef of inside_score_loss
    'lambda_side_vertex_code_loss': 1.0, # coef of vertex_code_loss
    "lambda_side_vertex_coord_loss": 1.0, # coef of vertex_coord_loss
    'max_train_img_size': 448, # max size of training images
    'max_predict_img_size': 448, # max size of the images to predict
    'ckpt_save_max': 10, # maximum of ckpt in dir
    'saved_model_file_path': './saved_model/', # dir of saved model
    'norm': 'BN', # normalization in feature merging branch
```

## [Evaluation Process](#contents)

The above python command will run in the background, you can view the results through the file output.eval.log. You will get the accuracy as following.
You can get loss, accuracy, recall, F1 score and the box vertices of an image.

- loss

```bash
# evaluate loss of the model
bash scripts/run_distribute_train_gpu.sh
```

- score

```bash
# evaluate loss of the model
bash scripts/run_distribute_train_gpu.sh
```

- prediction

```bash
# get prediction of an image
bash run_eval.sh 0_8-24_1012.ckpt pred ./demo/001.png
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

## performance

## [Training Performance](#contents)

The performance listed below are acquired with the default configurations in /src/config.py.
The Normalization of model training on Ascend is GN, the model training on GPU is used BN.
| Parameters           | single Ascend                            |  8 GPUs                            |
| -------------------- | ------------------------------------- |---------------------------------------------- |
| Model Version        | AdvancedEAST            | AdvancedEAST              |
| Resources            | Ascend 910 | Tesla V100S-PCIE 32G|
| MindSpore Version    | 1.1             |1.1                     |
| Dataset              | MTWI-2018           |MTWI-2018                 |
| Training Parameters  | epoch=18, batch_size = 8, lr=1e-3  |epoch=84, batch_size = 8, lr=1e-3  |
| Optimizer            | AdamWeightDecay             |AdamWeightDecay             |
| Loss Function        | QuadLoss |QuadLoss |
| Outputs              |  matrix with size of 3x64x64,3x96x96,3x112x112  |matrix with size of 3x64x64,3x96x96,3x112x112       |
| Loss                 | 0.1           |0.1           |
| Total Time           | 28 mins, 60 mins, 90 mins | 4.9 mins, 10.3 mins, 14.5 mins
| Checkpoints          | 173MB（.ckpt file）                |173MB（.ckpt file）                |

## [Evaluation Performance](#contents)

On the default
| Parameters  | single Ascend          | 8 GPUs                            |
| ------------------- | --------------------------- |--------------------------- |
| Model Version      | AdvancedEAST        |AdvancedEAST        |
| Resources        | Ascend 910         |Tesla V100S-PCIE 32G|
| MindSpore Version   | 1.1                 | 1.1                 |
| Dataset | 1000 images |1000 images |
| batch_size          |   8                        | 8                        |
| Outputs | precision, recall, F score |precision, recall, F score |
| performance | 94.35, 55.45, 66.31 | 92.53 55.49 66.01 |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
