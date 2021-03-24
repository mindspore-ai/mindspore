# Contents

- [DeepFM Description](#deepfm-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)  
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DeepFM Description](#contents)

Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods seem to have a strong bias towards low- or high-order interactions, or require expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and high-order feature interactions. The proposed model, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture.

[Paper](https://arxiv.org/abs/1703.04247):  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

# [Model Architecture](#contents)

DeepFM consists of two components. The FM component is a factorization machine, which is proposed in to learn feature interactions for recommendation. The deep component is a feed-forward neural network, which is used to learn high-order feature interactions.
The FM and deep component share the same input raw feature vector, which enables DeepFM to learn low- and high-order feature interactions simultaneously from the input raw features.

# [Dataset](#contents)

- [1] A dataset used in  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU, or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run distributed training example
  sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  [hccl tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

- running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file src/config.py

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run distributed training example
  sh scripts/run_distribute_train.sh 8 /dataset_path

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='GPU' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path/deepfm.ckpt
  ```

- running on CPU

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='CPU' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='CPU' > ms_log/eval_output.log 2>&1 &
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
└─deepfm
  ├─README.md
  ├─mindspore_hub_conf.md             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in Ascend or GPU
    ├─run_distribute_train.sh         # launch distributed training(8p) in Ascend
    ├─run_distribute_train_gpu.sh     # launch distributed training(8p) in GPU
    └─run_eval.sh                     # launch evaluating in Ascend or GPU
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─callback.py                     # define callback function
    ├─deepfm.py                       # deepfm network
    ├─dataset.py                      # create dataset for deepfm
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- train parameters

  ```help
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

  ```help
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

  ```shell
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

  ```log
  2020-05-27 15:26:29 epoch: 1 step: 41257, loss is 0.498953253030777
  2020-05-27 15:32:32 epoch: 2 step: 41257, loss is 0.45545706152915955
  ...
  ```

  The model checkpoint will be saved in the current directory.

- running on GPU

  To do.

### Distributed Training

- running on Ascend

  ```shell
  sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `log[X]/output.log`. The loss value are saved in loss.log file.

- running on GPU

  To do.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation.

  ```shell
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  The above python command will run in the background. You can view the results through the file "eval_output.log". The accuracy is saved in auc.log file.

  ```log
  {'result': {'AUC': 0.8057789065281104, 'eval_time': 35.64779996871948}}
  ```

- evaluation on dataset when running on GPU

  To do.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| Model Version              | DeepFM                                                      | To do                  |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G              | To do                  |
| uploaded Date              | 09/15/2020 (month/day/year)                                 | To do                  |
| MindSpore Version          | 1.0.0                                                 | To do                  |
| Dataset                    | [1]                                                         | To do                  |
| Training Parameters        | epoch=15, batch_size=1000, lr=1e-5                          | To do                  |
| Optimizer                  | Adam                                                        | To do                  |
| Loss Function              | Sigmoid Cross Entropy With Logits                           | To do                  |
| outputs                    | Accuracy                                                    | To do                  |
| Loss                       | 0.45                                                        | To do                  |
| Speed                      | 1pc: 8.16 ms/step;                                          | To do                  |
| Total time                 | 1pc: 90 mins;                                               | To do                  |
| Parameters (M)             | 16.5                                                        | To do                  |
| Checkpoint for Fine tuning | 190M (.ckpt file)                                           | To do                  |
| Scripts                    | [deepfm script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm) | To do                  |

### Inference Performance

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | DeepFM                      | To do                       |
| Resource            | Ascend 910                  | To do                       |
| Uploaded Date       | 05/27/2020 (month/day/year) | To do                       |
| MindSpore Version   | 0.3.0-alpha                 | To do                       |
| Dataset             | [1]                         | To do                       |
| batch_size          | 1000                        | To do                       |
| outputs             | accuracy                    | To do                       |
| Accuracy            | 1pc: 80.55%;                | To do                       |
| Model for inference | 190M (.ckpt file)           | To do                       |

# [Description of Random Situation](#contents)

We set the random seed before training in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
