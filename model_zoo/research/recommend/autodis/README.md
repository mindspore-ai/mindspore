# Contents

- [AutoDis Description](#AutoDis-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
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
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/autodis.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/autodis.ckpt
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
.
└─autodis
  ├─README.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in Ascend or GPU
    └─run_eval.sh                     # launch evaluating in Ascend or GPU
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─callback.py                     # define callback function
    ├─autodis.py                      # AutoDis network
    ├─dataset.py                      # create dataset for AutoDis
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- train parameters

  ```python
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

- eval parameters

  ```bash
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py \
    --dataset_path='dataset/train' \
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
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/autodis.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/autodis.ckpt
  ```

  The above python command will run in the background. You can view the results through the file "eval_output.log". The accuracy is saved in auc.log file.

  ```txt
  {'result': {'AUC': 0.8109881454077731, 'eval_time': 27.72783327102661s}}
  ```

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
| MindSpore Version   | 0.3.0-alpha                 |
| Dataset             | [1]                         |
| batch_size          | 1000                        |
| outputs             | accuracy                    |
| AUC            | 1pc: 0.8112;                |
| Model for inference | 191M (.ckpt file)           |

# [Description of Random Situation](#contents)

We set the random seed before training in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
