# Contents

- [FAT-DeepFFM Description](#deepfm-description)
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
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FAT-DeepFFM Description](#contents)

Click-through rate estimation is a very important part of computing advertising and recommendation systems. Meanwhile, CTR models often use some commonly used methods in other fields, such as computer vision and natural language processing. The most common one is the Attention mechanism. Use the Attention mechanism to pick out the most important features from the list and filter out the irrelevant ones. The attention mechanism is combined with CTR prediction model of deep learning.

[Paper](https://arxiv.org/abs/1905.06336): Junlin Zhang , Tongwen Huang , Zhiqi Zhang FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine

# [Model Architecture](#contents)

Fat - DeepFFM consists of three parts. The FFM component is a factorization machine that is proposed to learn feature interactions for recommendation. The depth component is a feedforward neural network for learning higher-order feature interactions, and the attention part is the self-attention mechanism of features. The output of the initial feature from attention is then entered into the depth module. FAT-deepffm can simultaneously learn low-order and high-order feature interactions from the input original feature.

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

- Download the Dataset

  > Please refer to [1] to obtain the download link

  ```bash
  mkdir -p data/ && cd data/
  wget DATA_LINK
  tar -zxvf dac.tar.gz
  ```

- Use this script to preprocess the data. This may take about one hour and the generated mindrecord data is under data/mindrecord.

  ```bash
  python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
  ```

- running on Ascend

  ```shell
  # run training example
  python train.py \
    --dataset_path='data/mindrecord' \
    --ckpt_path='./checkpoint/Fat-DeepFFM' \
    --eval_file_name='./auc.log' \
    --loss_file_name='./loss.log' \
    --device_target='Ascend' \
    --do_eval=True > output.log 2>&1 &

  # run distributed training example
   bash scripts/run_distribute_train.sh  /dataset_path 8  scripts/hccl_8p.json False

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/mindrecord' \
    --ckpt_path='./checkpoint/Fat-DeepFFM.ckpt'\
    --device_target = 'Ascend'\
    --device_id=0  > eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh  0 Ascend  /dataset_path  /ckpt_path
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  [hccl tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
└─Fat-deepffm
  ├─README.md
  ├─asecend310                        # C++ running module
  ├─scripts
    ├─run_alone_train.sh              # launch standalone training(1p) in Ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in Ascend
    └─run_eval.sh                     # launch evaluating in Ascend
  ├─src
    ├─config.py                       # parameter configuration
    ├─callback.py                     # define callback function
    ├─fat-deepfm.py                   # fat-deepffm network
    ├─lr_generator.py                 # generative learning rate
    ├─metrics.py                      # verify the model
    ├─dataset.py                      # create dataset for deepfm
  ├─eval.py                           # eval net
  ├─eval310.py                        # infer 310 net
  ├─GetDatasetBinary.py               # get binary dataset
  ├─export.py                         # export net
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
  --ckpt_path CHECKPOINT_PATH
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
    --dataset_path='/data/' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='./auc.log' \
    --loss_file_name='./loss.log' \
    --device_target='Ascend' \
    --do_eval=True > output.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `ms_log/output.log`.

  After training, you'll get some checkpoint files under `./checkpoint` folder by default. The loss value are saved in loss.log file.

  ```log
  2021-06-19 21:59:10 epoch: 1 step: 5166, loss is 0.46262410283088684
  2021-06-19 22:12:13 epoch: 2 step: 5166, loss is 0.4792023301124573
  2021-06-19 22:21:03 epoch: 3 step: 5166, loss is 0.4666571617126465
  2021-06-19 22:29:54 epoch: 4 step: 5166, loss is 0.44029417634010315
  ...
  ```

  The model checkpoint will be saved in the current directory.

### Distributed Training

- running on Ascend

  ```shell
   bash scripts/run_distribute_train.sh  /dataset_path 8  scripts/hccl_8p.json False
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `log[X]/output.log`. The loss value are saved in loss.log file.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation.

  ```shell
  python eval.py \
    --dataset_path=' /dataset_path' \
    --checkpoint_path='/ckpt_path' \
    --device_id=0 \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
   bash scripts/run_eval.sh  0 Ascend  /dataset_path  /ckpt_path
  ```

  The above python command will run in the background. You can view the results through the file "eval_output.log". The accuracy is saved in auc.log file.

  ```log
  {'AUC': 0.8091001899667086}
  ```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.
- `DATASET_PATH` is path that contains the mindrecord dataset.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'AUC': 0.8088441692761583
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | Fat-DeepFFM                                                      |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8                |
| uploaded Date              | 09/15/2020 (month/day/year)                                 |
| MindSpore Version          | 1.2.0                                                 |
| Dataset                    | Criteo                                                       |
| Training Parameters        | epoch=30, batch_size=1000, lr=1e-4                          |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy With Logits                           |
| outputs                    | AUC                                                    |
| Loss                       | 0.45                                                        |
| Speed                      | 1pc: 8.16 ms/step;                                          |
| Total time                 | 1pc: 4 hours;                                               |
| Parameters (M)             | 560.34                                                        |
| Checkpoint for Fine tuning | 87.65M (.ckpt file)                                           |
| Scripts                    | [deepfm script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/recommend/Fat-DeepFFM) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | DeepFM                      |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 06/20/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | Criteo                      |
| batch_size          | 1000                        |
| outputs             | AUC                         |
| AUC                 | 1pc: 80.90%;                |
| Model for inference | 87.65M (.ckpt file)         |

# [Description of Random Situation](#contents)

We set the random seed before training in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
