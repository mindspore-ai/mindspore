# Contents

- [DenseNet Description](#densenet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
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
        - [Training accuracy results](#training-accuracy-results)
        - [Training performance results](#training-performance-results)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DenseNet Description](#contents)

DenseNet is a convolution based neural network for the task of image classification. The paper describing the model can be found [here](https://arxiv.org/abs/1608.06993). HuaWei’s DenseNet is a implementation on [MindSpore](https://www.mindspore.cn/).

The repository also contains scripts to launch training and inference routines.

# [Model Architecture](#contents)

DenseNet supports two kinds of implementations: DenseNet100 and DenseNet121, where the number represents number of layers in the network.

DenseNet121 builds on 4 densely connected block and DenseNet100 builds on 3. In every dense block, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Concatenation is used. Each layer is receiving a “collective knowledge” from all preceding layers.

# [Dataset](#contents)

Dataset used in DenseNet121: ImageNet

The default configuration of the Dataset are as follows:

- Training Dataset preprocess:
    - Input size of images is 224\*224
    - Range (min, max) of respective size of the original size to be cropped is (0.08, 1.0)
    - Range (min, max) of aspect ratio to be cropped is (0.75, 1.333)
    - Probability of the image being flipped set to 0.5
    - Randomly adjust the brightness, contrast, saturation (0.4, 0.4, 0.4)
    - Normalize the input image with respect to mean and standard deviation

- Test Dataset preprocess:
    - Input size of images is 224\*224 (Resize to 256\*256 then crops images at the center)
    - Normalize the input image with respect to mean and standard deviation

Dataset used in DenseNet100: Cifar-10

The default configuration of the Dataset are as follows:

- Training Dataset preprocess:
    - Input size of images is 32\*32
    - Randomly cropping is applied to the image with padding=4
    - Probability of the image being flipped set to 0.5
    - Randomly adjust the brightness, contrast, saturation (0.4, 0.4, 0.4)
    - Normalize the input image with respect to mean and standard deviation

- Test Dataset preprocess:
    - Input size of images is 32\*32
    - Normalize the input image with respect to mean and standard deviation

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

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
  python train.py --net [NET_NAME] --dataset [DATASET_NAME] --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 &

  # run distributed training example
  sh scripts/run_distribute_train.sh 8 rank_table.json [NET_NAME] [DATASET_NAME] /PATH/TO/DATASET /PATH/TO/PRETRAINED_CKPT

  # run evaluation example
  python eval.py --net [NET_NAME] --dataset [DATASET_NAME] --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT > eval.log 2>&1 &
  OR
  sh scripts/run_distribute_eval.sh 8 rank_table.json [NET_NAME] [DATASET_NAME] /PATH/TO/DATASET /PATH/TO/CHECKPOINT
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- running on GPU

  For running on GPU, please change `platform` from `Ascend` to `GPU`

  ```python
  # run training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py --net=[NET_NAME] --dataset=[DATASET_NAME] --data_dir=[DATASET_PATH] --is_distributed=0 --device_target='GPU' > train.log 2>&1 &

  # run distributed training example
  sh run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7  [NET_NAME] [DATASET_NAME] [DATASET_PATH]

  # run evaluation example
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --data_dir=[DATASET_PATH] --device_target='GPU' --pretrained=[CHECKPOINT_PATH] > eval.log 2>&1 &
  OR
  sh run_distribute_eval_gpu.sh 1 0  [NET_NAME] [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── densenet
        ├── README.md                    // descriptions about densenet
        ├── scripts
        │   ├── run_distribute_train.sh             // shell script for distributed on Ascend
        │   ├── run_distribute_train_gpu.sh             // shell script for distributed on GPU
        │   ├── run_distribute_eval.sh              // shell script for evaluation on Ascend
        │   ├── run_distribute_eval_gpu.sh              // shell script for evaluation on GPU
        ├── src
        │   ├── datasets             // dataset processing function
        │   ├── losses
        │       ├──crossentropy.py            // densenet loss function
        │   ├── lr_scheduler
        │       ├──lr_scheduler.py            // densenet learning rate schedule function
        │   ├── network
        │       ├──densenet.py            // densenet architecture
        │   ├──optimizers            // densenet optimize function
        │   ├──utils
        │       ├──logging.py            // logging function
        │       ├──var_init.py            // densenet variable init function
        │   ├── config.py             // network config
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
```

## [Script Parameters](#contents)

You can modify the training behaviour through the various flags in the `train.py` script. Flags in the `train.py` script are as follows:

```python
  --data_dir              train data dir
  --num_classes           num of classes in dataset(default:1000 for densenet121; 10 for densenet100)
  --image_size            image size of the dataset
  --per_batch_size        mini-batch size (default: 32 for densenet121; 64 for densenet100) per gpu
  --pretrained            path of pretrained model
  --lr_scheduler          type of LR schedule: exponential, cosine_annealing
  --lr                    initial learning rate
  --lr_epochs             epoch milestone of lr changing
  --lr_gamma              decrease lr by a factor of exponential lr_scheduler
  --eta_min               eta_min in cosine_annealing scheduler
  --T_max                 T_max in cosine_annealing scheduler
  --max_epoch             max epoch num to train the model
  --warmup_epochs         warmup epoch(when batchsize is large)
  --weight_decay          weight decay (default: 1e-4)
  --momentum              momentum(default: 0.9)
  --label_smooth          whether to use label smooth in CE
  --label_smooth_factor   smooth strength of original one-hot
  --log_interval          logging interval(default:100)
  --ckpt_path             path to save checkpoint
  --ckpt_interval         the interval to save checkpoint
  --is_save_on_master     save checkpoint on master or all rank
  --is_distributed        if multi device(default: 1)
  --rank                  local rank of distributed(default: 0)
  --group_size            world size of distributed(default: 1)
```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py --net [NET_NAME] --dataset [DATASET_NAME] --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 &
  ```

  The python command above will run in the background, The log and model checkpoint will be generated in `output/202x-xx-xx_time_xx_xx_xx/`. The loss value of training DenseNet121 on ImageNet will be achieved as follows:

  ```shell
  2020-08-22 16:58:56,617:INFO:epoch[0], iter[5003], loss:4.367, mean_fps:0.00 imgs/sec
  2020-08-22 16:58:56,619:INFO:local passed
  2020-08-22 17:02:19,920:INFO:epoch[1], iter[10007], loss:3.193, mean_fps:6301.11 imgs/sec
  2020-08-22 17:02:19,921:INFO:local passed
  2020-08-22 17:05:43,112:INFO:epoch[2], iter[15011], loss:3.096, mean_fps:6304.53 imgs/sec
  2020-08-22 17:05:43,113:INFO:local passed
  ...
  ```

- running on GPU

  ```python
  export CUDA_VISIBLE_DEVICES=0
  python train.py --net [NET_NAME] --dataset [DATASET_NAME] --data_dir=[DATASET_PATH] --is_distributed=0 --device_target='GPU' > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder `./ckpt_0/` by default.

- running on CPU

  ```python
  python train.py --net=[NET_NAME] --dataset=[DATASET_NAME] --data_dir=[DATASET_PATH] --is_distributed=0 --device_target='CPU' > train.log 2>&1 &
  ```

  The python command above will run in the background, The log and model checkpoint will be generated in `output/202x-xx-xx_time_xx_xx_xx/`.

### Distributed Training

- running on Ascend

  ```bash
  sh scripts/run_distribute_train.sh 8 rank_table.json [NET_NAME]  [DATASET_NAME] /PATH/TO/DATASET /PATH/TO/PRETRAINED_CKPT
  ```

  The above shell script will run distribute training in the background. You can view the results log and model checkpoint through the file `train[X]/output/202x-xx-xx_time_xx_xx_xx/`. The loss value of training DenseNet121 on ImageNet will be achieved as follows:

  ```log
  2020-08-22 16:58:54,556:INFO:epoch[0], iter[5003], loss:3.857, mean_fps:0.00 imgs/sec
  2020-08-22 17:02:19,188:INFO:epoch[1], iter[10007], loss:3.18, mean_fps:6260.18 imgs/sec
  2020-08-22 17:05:42,490:INFO:epoch[2], iter[15011], loss:2.621, mean_fps:6301.11 imgs/sec
  2020-08-22 17:09:05,686:INFO:epoch[3], iter[20015], loss:3.113, mean_fps:6304.37 imgs/sec
  2020-08-22 17:12:28,925:INFO:epoch[4], iter[25019], loss:3.29, mean_fps:6303.07 imgs/sec
  2020-08-22 17:15:52,167:INFO:epoch[5], iter[30023], loss:2.865, mean_fps:6302.98 imgs/sec
  ...
  ...
  ```

- running on GPU

  ```bash
  cd scripts
  sh run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 [NET_NAME] [DATASET_NAME] [DATASET_PATH]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train/train.log`.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on Ascend

  running the command below for evaluation.

  ```python
  python eval.py --net [NET_NAME] --dataset [DATASET_NAME] --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT > eval.log 2>&1 &
  OR
  sh scripts/run_distribute_eval.sh 8 rank_table.json [NET_NAME] [DATASET_NAME] /PATH/TO/DATASET /PATH/TO/CHECKPOINT
  ```

  The above python command will run in the background. You can view the results through the file "output/202x-xx-xx_time_xx_xx_xx/202x_xxxx.log". The accuracy of evaluating DenseNet121 on the test dataset of ImageNet will be as follows:

  ```log
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top1_correct=37657, tot=49920, acc=75.43%
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top5_correct=46224, tot=49920, acc=92.60%
  ```

- evaluation on GPU

  running the command below for evaluation.

  ```python
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --data_dir=[DATASET_PATH] --device_target='GPU' --pretrained=[CHECKPOINT_PATH] > eval.log 2>&1 &
  OR
  sh run_distribute_eval_gpu.sh 1 0 [NET_NAME] [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log". The accuracy of evaluating DenseNet121 on the test dataset of ImageNet will be as follows:

  ```log
  2021-02-04 14:20:50,551:INFO:after allreduce eval: top1_correct=37637, tot=49984, acc=75.30%
  2021-02-04 14:20:50,551:INFO:after allreduce eval: top5_correct=46370, tot=49984, acc=92.77%
  ```

  The accuracy of evaluating DenseNet100 on the test dataset of Cifar-10 will be as follows:

  ```log
  2021-03-12 18:04:07,893:INFO:after allreduce eval: top1_correct=9536, tot=9984, acc=95.51%
  ```

- evaluation on CPU

  running the command below for evaluation.

  ```python
  python eval.py --net=[NET_NAME] --dataset=[DATASET_NAME] --data_dir=[DATASET_PATH] --device_target='CPU' --pretrained=[CHECKPOINT_PATH] > eval.log 2>&1 &
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log".  The accuracy of evaluating DenseNet100 on the test dataset of Cifar-10 will be as follows:

  ```log
  2021-03-18 09:06:43,247:INFO:after allreduce eval: top1_correct=9492, tot=9984, acc=95.07%
  ```

# [Model Description](#contents)

## [Performance](#contents)

### DenseNet121

### Training accuracy results

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | DenseNet121               | DenseNet121               |
| Resource            | Ascend 910                  | Tesla V100-PCIE             |
| Uploaded Date       | 09/15/2020 (month/day/year) | 01/27/2021 (month/day/year) |
| MindSpore Version   | 1.0.0                       | 1.1.0                       |
| Dataset             | ImageNet                    | ImageNet                    |
| epochs              | 120                         | 120                         |
| outputs             | probability                 | probability                 |
| accuracy            | Top1:75.13%; Top5:92.57%    | Top1:75.30%; Top5:92.77%    |

### Training performance results

| Parameters          | Ascend                      | GPU                          |
| ------------------- | --------------------------- | ---------------------------- |
| Model Version       | DenseNet121              | DenseNet121               |
| Resource            | Ascend 910                  | Tesla V100-PCIE              |
| Uploaded Date       | 09/15/2020 (month/day/year) | 02/04/2021 (month/day/year)  |
| MindSpore Version   | 1.0.0                       | 1.1.1                        |
| Dataset             | ImageNet                    | ImageNet                     |
| batch_size          | 32                          | 32                           |
| outputs             | probability                 | probability                  |
| speed               | 1pc:760 img/s;8pc:6000 img/s| 1pc:161 img/s;8pc:1288 img/s |

### DenseNet100

### Training performance

| Parameters          | GPU                          |
| ------------------- | ---------------------------- |
| Model Version       | DenseNet100               |
| Resource            | Tesla V100-PCIE              |
| Uploaded Date       | 03/18/2021 (month/day/year)  |
| MindSpore Version   | 1.2.0                        |
| Dataset             | Cifar-10                     |
| batch_size          | 64                           |
| epochs              | 300                         |
| outputs             | probability                  |
| accuracy            | 95.31%                  |
| speed               | 1pc: 600.07 img/sec    |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
