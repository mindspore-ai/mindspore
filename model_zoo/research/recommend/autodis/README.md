# Contents

- [Contents](#contents)
- [AutoDis Description](#autodis-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [AutoDis Description](#contents)

The common methods for numerical feature embedding are Normalization and Discretization. The former shares a single embedding for intra-field features and the latter transforms the features into categorical form through various discretization approaches. However, the first approach surfers from low capacity and the second one limits performance as well because the discretization rule cannot be optimized with the ultimate goal of CTR model.
To fill the gap of representing numerical features, in this paper, we propose AutoDis, a framework that discretizes features in numerical fields automatically and is optimized with CTR models in an end-to-end manner. Specifically, we introduce a set of meta-embeddings for each numerical field to model the relationship among the intra-field features and propose an automatic differentiable discretization and aggregation approach to capture the correlations between the numerical features and meta-embeddings.  AutoDis is a valid framework to work with various popular deep CTR models  and  is  able  to  improve  the  recommendation  performance significantly.

[Paper](https://arxiv.org/abs/2012.08986):  Huifeng Guo*, Bo Chen*, Ruiming Tang, Zhenguo Li, Xiuqiang He. AutoDis: Automatic Discretization for Embedding Numerical Features in CTR Prediction

# [Model Architecture](#contents)

AutoDis leverages a set of meta-embeddings for each numerical field, which are shared among all the intra-field feature values. Meta-embeddings learn the relationship across different feature values in this field with a manageable number of embedding parameters. Utilizing meta-embedding is able to avoid explosive embedding parameters introduced by assigning each numerical feature with an independent embedding simply. Besides, the embedding of a numerical feature is designed as a differentiable aggregation over the shared meta-embeddings, so that the discretization of numerical features can be optimized with the ultimate goal of deep CTR models in an end-to-end manner.

# [Dataset](#contents)

- [1] A dataset Criteo used in  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/r1.3/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # run training example
  python train.py \
    --train_data_dir='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run evaluation example
  python eval.py \
    --test_data_dir='dataset/test' \
    --checkpoint_path='./checkpoint/autodis.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /test_data_dir /checkpoint_path/autodis.ckpt
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- running on ModelArts

  If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

    - Training with single cards on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/autodis" on the website UI interface.
    # (4) Set the startup file to /{path}/autodis/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/autodis/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
    ```

    - evaluating with single card on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/autodis" on the website UI interface.
    # (4) Set the startup file to /{path}/autodis/eval.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/autodis/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “checkpoint_path: ./{path}/*.ckpt”('checkpoint_path' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “checkpoint_path=./{path}/*.ckpt”('checkpoint_path' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of a single card.
    # (10) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
.
└─autodis
  ├─README.md                         # descriptions of warpctc
  ├─ascend310_infer                   # application for 310 inference
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in Ascend or GPU
    ├─run_infer_310.sh                # launch 310infer
    └─run_eval.sh                     # launch evaluating in Ascend or GPU
  ├─src
    ├─model_utils
      ├─config.py                     # parsing parameter configuration file of "*.yaml"
      ├─device_adapter.py             # local or ModelArts training
      ├─local_adapter.py              # get related environment variables in local training
      └─moxing_adapter.py             # get related environment variables in ModelArts training
    ├─__init__.py                     # python init file
    ├─callback.py                     # define callback function
    ├─autodis.py                      # AutoDis network
    └─dataset.py                      # create dataset for AutoDis
  ├─default_config.yaml               # parameter configuration
  ├─eval.py                           # eval script
  ├─export.py                         # export checkpoint file into air/mindir
  ├─mindspore_hub_conf.py             # mindspore hub interface
  ├─postprocess.py                    # 310infer postprocess script
  ├─preprocess.py                     # 310infer preprocess script
  └─train.py                          # train script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `default_config.yaml`

- Parameters that can be modified at the terminal

  ```bash
  # Train
  train_data_dir: ''                  # train dataset path
  ckpt_path: 'ckpts'                  # the folder path to save '*.ckpt' files. Relative path.
  eval_file_name: "./auc.log"         # file path to record accuracy
  loss_file_name: "./loss.log"        # file path to record loss
  do_eval: "True"                     # whether do eval while training, default is 'True'.
  # Test
  test_data_dir: ''                   # test dataset path
  checkpoint_path: ''                 # the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.
  # Export
  batch_size: 16000                   # batch_size for exported model.
  ckpt_file: ''                       # the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.
  file_name: "autodis"                # output file name.
  file_format: "AIR"                  # output file format, you can choose from AIR or MINDIR, default is AIR"
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py \
    --train_data_dir='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `ms_log/output.log`.

  After training, you'll get some checkpoint files under `./checkpoint` folder by default. The loss value are saved in loss.log file.

  ```txt
  2020-12-10 14:58:04 epoch: 1 step: 41257, loss is 0.44559600949287415
  2020-12-10 15:06:59 epoch: 2 step: 41257, loss is 0.4370603561401367
  ...
  ```

  The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation.

  ```python
  python eval.py \
    --test_data_dir='dataset/test' \
    --checkpoint_path='./checkpoint/autodis.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /test_data_dir /checkpoint_path/autodis.ckpt
  ```

  The above python command will run in the background. You can view the results through the file "eval_output.log". The accuracy is saved in auc.log file.

  ```txt
  {'result': {'AUC': 0.8109881454077731, 'eval_time': 27.72783327102661s}}
  ```

## Inference Process

### [Export MindIR](#contents)

- Export on local

  ```shell
  # The ckpt_file parameter is required, `EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
  python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

  ```python
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/autodis" on the website UI interface.
  # (4) Set the startup file to /{path}/autodis/export.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/autodis/default_config.yaml.
  #         1. Set ”enable_modelarts: True“
  #         2. Set “ckpt_file: ./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Set ”file_name: autodis“
  #         4. Set ”file_format：AIR“(you can choose from AIR or MINDIR)
  #     b. adding on the website UI interface.
  #         1. Add ”enable_modelarts=True“
  #         2. Add “ckpt_file=./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Add ”file_name=autodis“
  #         4. Add ”file_format=AIR“(you can choose from AIR or MINDIR)
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # You will see autodis.air under "Output file path".
  ```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | AutoDis                                                      |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G              |
| uploaded Date              | 12/12/2020 (month/day/year)                                 |
| MindSpore Version          | 1.1.0                                                 |
| Dataset                    | [1]                                                         |
| Training Parameters        | epoch=15, batch_size=1000, lr=1e-5                          |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy With Logits                           |
| outputs                    | Accuracy                                                    |
| Loss                       | 0.42                                                        |
| Speed                      | 1pc: 8.16 ms/step;                                          |
| Total time                 | 1pc: 90 mins;                                               |
| Parameters (M)             | 16.5                                                        |
| Checkpoint for Fine tuning | 191M (.ckpt file)                                           |
| Scripts                    | [AutoDis script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/recommend/autodis) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | AutoDis                      |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/12/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | [1]                         |
| batch_size          | 1000                        |
| outputs             | accuracy                    |
| AUC            | 1pc: 0.8112;                |
| Model for inference | 191M (.ckpt file)           |

# [Description of Random Situation](#contents)

We set the random seed before training in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
