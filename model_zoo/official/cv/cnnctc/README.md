# Contents

- [CNNCTC Description](#CNNCTC-description)
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
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CNNCTC Description](#contents)

This paper proposes three major contributions to addresses scene text recognition (STR).
First, we examine the inconsistencies of training and evaluation datasets, and the performance gap results from inconsistencies.
Second, we introduce a unified four-stage STR framework that most existing STR models fit into.
Using this framework allows for the extensive evaluation of previously proposed STR modules and the discovery of previously
unexplored module combinations. Third, we analyze the module-wise contributions to performance in terms of accuracy, speed,
and memory demand, under one consistent set of training and evaluation datasets. Such analyses clean up the hindrance on the current
comparisons to understand the performance gain of the existing modules.
[Paper](https://arxiv.org/abs/1904.01906): J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, “What is wrong with scene text recognition model comparisons? dataset and model analysis,” ArXiv, vol. abs/1904.01906, 2019.

# [Model Architecture](#contents)

This is an example of training CNN+CTC model for text recognition on MJSynth and SynthText dataset with MindSpore.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

The [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](https://github.com/ankush-me/SynthText) dataset are used for model training. The [The IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) dataset is used for evaluation.

- step 1:

All the datasets have been preprocessed and stored in .lmdb format and can be downloaded [**HERE**](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt).

- step 2:

Uncompress the downloaded file, rename the MJSynth dataset as MJ, the SynthText dataset as ST and the IIIT dataset as IIIT.

- step 3:

Move above mentioned three datasets into `cnnctc_data` folder, and the structure should be as below:

```text
|--- CNNCTC/
    |--- cnnctc_data/
        |--- ST/
            data.mdb
            lock.mdb
        |--- MJ/
            data.mdb
            lock.mdb
        |--- IIIT/
            data.mdb
            lock.mdb

    ......
```

- step 4:

Preprocess the dataset by running:

```bash
python src/preprocess_dataset.py
```

This takes around 75 minutes.

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）

    - Prepare hardware environment with Ascend or GPU processor.
- Framework

    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)

    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

- Install dependencies:

```bash
pip install lmdb
pip install Pillow
pip install tqdm
pip install six
```

```default_config.yaml

TRAIN_DATASET_PATH: /home/DataSet/MJ-ST-IIIT/ST-MJ/
TRAIN_DATASET_INDEX_PATH: /home/DataSet/MJ-ST-IIIT/st_mj_fixed_length_index_list.pkl
TEST_DATASET_PATH: /home/DataSet/MJ-ST-IIIT/IIIT5K_3000

Modify the parameters according to the actual path
```

- Standalone Ascend Training:

```bash
bash scripts/run_standalone_train_ascend.sh $DEVICE_ID $PRETRAINED_CKPT(options)
# example: bash scripts/run_standalone_train_ascend.sh 0
```

- Standalone GPU Training:

```bash
bash scripts/run_standalone_train_gpu.sh $PRETRAINED_CKPT(options)
```

- Distributed Ascend Training:

```bash
bash scripts/run_distribute_train_ascend.sh $RANK_TABLE_FILE $PRETRAINED_CKPT(options)
# example: bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json
```

- Distributed GPU Training:

```bash
bash scripts/run_distribute_train_gpu.sh $PRETRAINED_CKPT(options)
```

- Ascend Evaluation:

```bash
bash scripts/run_eval_ascend.sh $DEVICE_ID $TRAINED_CKPT
# example: scripts/run_eval_ascend.sh 0 /home/model/cnnctc/ckpt/CNNCTC-1_8000.ckpt
```

- GPU Evaluation:

```bash
bash scripts/run_eval_gpu.sh $TRAINED_CKPT
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```text
|--- CNNCTC/
    |---README.md    // descriptions about cnnctc
    |---README_cn.md    // descriptions about cnnctc
    |---default_config.yaml   // config file
    |---train.py    // train scripts
    |---eval.py    // eval scripts
    |---export.py    // export scripts
    |---preprocess.py     // preprocess scripts
    |---postprocess.py    // postprocess scripts
    |---ascend310_infer    // application for 310 inference
    |---scripts
        |---run_infer_310.sh    // shell script for infer on ascend310
        |---run_standalone_train_ascend.sh    // shell script for standalone on ascend
        |---run_standalone_train_gpu.sh    // shell script for standalone on gpu
        |---run_distribute_train_ascend.sh    // shell script for distributed on ascend
        |---run_distribute_train_gpu.sh    // shell script for distributed on gpu
        |---run_eval_ascend.sh    // shell script for eval on ascend
    |---src
        |---__init__.py    // init file
        |---cnn_ctc.py    // cnn_ctc network
        |---callback.py    // loss callback file
        |---dataset.py    // process dataset
        |---util.py    // routine operation
        |---preprocess_dataset.py    // preprocess dataset
        |--- model_utils
            |---config.py             // Parameter config
            |---moxing_adapter.py     // modelarts device configuration
            |---device_adapter.py     // Device Config
            |---local_adapter.py      // local device config
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `default_config.yaml`.

Arguments:

- `--CHARACTER`: Character labels.
- `--NUM_CLASS`: The number of classes including all character labels and the <blank> label for CTCLoss.
- `--HIDDEN_SIZE`: Model hidden size.
- `--FINAL_FEATURE_WIDTH`: The number of features.
- `--IMG_H`： The height of input image.
- `--IMG_W`： The width of input image.
- `--TRAIN_DATASET_PATH`： The path to training dataset.
- `--TRAIN_DATASET_INDEX_PATH`： The path to training dataset index file which determines the order .
- `--TRAIN_BATCH_SIZE`： Training batch size. The batch size and index file must ensure input data is in fixed shape.
- `--TRAIN_DATASET_SIZE`： Training dataset size.
- `--TEST_DATASET_PATH`： The path to test dataset.
- `--TEST_BATCH_SIZE`： Test batch size.
- `--TRAIN_EPOCHS`：Total training epochs.
- `--CKPT_PATH`：The path to model checkpoint file, can be used to resume training and evaluation.
- `--SAVE_PATH`：The path to save model checkpoint file.
- `--LR`：Learning rate for standalone training.
- `--LR_PARA`：Learning rate for distributed training.
- `--MOMENTUM`：Momentum.
- `--LOSS_SCALE`：Loss scale to prevent gradient underflow.
- `--SAVE_CKPT_PER_N_STEP`：Save model checkpoint file per N steps.
- `--KEEP_CKPT_MAX_NUM`：The maximum number of saved model checkpoint file.

## [Training Process](#contents)

### Training

- Standalone Ascend Training:

```bash
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [PRETRAINED_CKPT(options)]
# example: bash scripts/run_standalone_train_ascend.sh 0
```

Results and checkpoints are written to `./train` folder. Log can be found in `./train/log` and loss values are recorded in `./train/loss.log`.

`$PRETRAINED_CKPT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

- Distributed Ascend Training:

```bash
bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT(options)]
# example: bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json
```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

Results and checkpoints are written to `./train_parallel_{i}` folder for device `i` respectively.
 Log can be found in `./train_parallel_{i}/log_{i}.log` and loss values are recorded in `./train_parallel_{i}/loss.log`.

`$RANK_TABLE_FILE` is needed when you are running a distribute task on ascend.
`$PATH_TO_CHECKPOINT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

### Training Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss.log.

```text
# distribute training result(8p)
epoch: 1 step: 1 , loss is 76.25, average time per step is 0.235177839748392712
epoch: 1 step: 2 , loss is 73.46875, average time per step is 0.25798572540283203
epoch: 1 step: 3 , loss is 69.46875, average time per step is 0.229678678512573
epoch: 1 step: 4 , loss is 64.3125, average time per step is 0.23512671788533527
epoch: 1 step: 5 , loss is 58.375, average time per step is 0.23149147033691406
epoch: 1 step: 6 , loss is 52.7265625, average time per step is 0.2292975425720215
...
epoch: 1 step: 8689 , loss is 9.706798802612482, average time per step is 0.2184656601312549
epoch: 1 step: 8690 , loss is 9.70612545289855, average time per step is 0.2184725407765116
epoch: 1 step: 8691 , loss is 9.70695776049204, average time per step is 0.21847309686135555
epoch: 1 step: 8692 , loss is 9.707279624277456, average time per step is 0.21847339290613375
epoch: 1 step: 8693 , loss is 9.70763437950938, average time per step is 0.2184720295013031
epoch: 1 step: 8694 , loss is 9.707695425072046, average time per step is 0.21847410284595573
epoch: 1 step: 8695 , loss is 9.708408273381295, average time per step is 0.21847338271072345
epoch: 1 step: 8696 , loss is 9.708703753591953, average time per step is 0.2184726025560777
epoch: 1 step: 8697 , loss is 9.709536406025824, average time per step is 0.21847212061114694
epoch: 1 step: 8698 , loss is 9.708542263610315, average time per step is 0.2184715309307257
```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training dpn on modelarts :
#  Data set storage method

#  ├── CNNCTC_Data                                              # dataset dir
#    ├──train                                                   # train dir
#      ├── ST_MJ                                                # train dataset dir
#        ├── data.mdb                                           # data file
#        ├── lock.mdb
#      ├── st_mj_fixed_length_index_list.pkl
#    ├── eval                                                   # eval dir
#      ├── IIIT5K_3000                                          # eval dataset dir
#      ├── checkpoint                                           # checkpoint dir

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True"
#          set "run_distribute=True"
#          set "TRAIN_DATASET_PATH=/cache/data/ST_MJ/"
#          set "TRAIN_DATASET_INDEX_PATH=/cache/data/st_mj_fixed_length_index_list.pkl"
#          set "SAVE_PATH=/cache/train/checkpoint"
#
#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/cnnctc"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../CNNCTC_Data/train"(choices CNNCTC_Data/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#        a.set "enable_modelarts=True"
#          set "TEST_DATASET_PATH=/cache/data/IIIT5K_3000/"
#          set "CHECKPOINT_PATH=/cache/data/checkpoint/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/cnnctc"。
# (5) Set the model's startup file on the modelarts interface "train.py" 。
# (6) Set the data path of the model on the modelarts interface ".../CNNCTC_Data/train"(choices CNNCTC_Data/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

- Standalone GPU Training:

```bash
bash scripts/run_standalone_train_gpu.sh [PRETRAINED_CKPT(options)]
```

Results and checkpoints are written to `./train` folder. Log can be found in `./train/log` and loss values are recorded in `./train/loss.log`.

`$PRETRAINED_CKPT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

- Distributed GPU Training:

```bash
bash scripts/run_distribute_train_gpu.sh [PRETRAINED_CKPT(options)]
```

Results and checkpoints are written to `./train_parallel` folder with model checkpoints in ckpt_{i} directories.
Log can be found in `./train_parallel/log` and loss values are recorded in `./train_parallel/loss.log`.

## [Evaluation Process](#contents)

### Evaluation

- Ascend Evaluation:

```bash
bash scripts/run_eval_ascend.sh [DEVICE_ID] [TRAINED_CKPT]
# example: scripts/run_eval_ascend.sh 0 /home/model/cnnctc/ckpt/CNNCTC-1_8000.ckpt
```

The model will be evaluated on the IIIT dataset, sample results and overall accuracy will be printed.

- GPU Evaluation:

```bash
bash scripts/run_eval_gpu.sh [TRAINED_CKPT]
```

## [Inference process](#contents)

### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --TEST_BATCH_SIZE [BATCH_SIZE]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].
`BATCH_SIZE` current batch_size can only be set to 1.

- Export MindIR on Modelarts

```Modelarts
Export MindIR example on ModelArts
Data storage method is the same as training
# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
#       a. set "enable_modelarts=True"
#          set "file_name=cnnctc"
#          set "file_format=MINDIR"
#          set "ckpt_file=/cache/data/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted
# (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/cnnctc"。
# (4) Set the model's startup file on the modelarts interface "export.py" 。
# (5) Set the data path of the model on the modelarts interface ".../CNNCTC_Data/eval/checkpoint"(choices CNNCTC_Data/eval/checkpoint Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. CNNCTC only support CPU mode .
- `DEVICE_ID` is optional, default value is 0.

### Result

- Ascend Result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'Accuracy': 0.8642
```

- GPU result

Inference result is saved in ./eval/log, you can find result like this.

```bash
accuracy:  0.8533
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | CNNCTC                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |
| uploaded Date              | 09/28/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                      |
| Dataset                    |  MJSynth,SynthText                                                  |
| Training Parameters        | epoch=3,  batch_size=192          |
| Optimizer                  | RMSProp                                                         |
| Loss Function              | CTCLoss                                      |
| Speed                      | 1pc: 250 ms/step;  8pcs: 260 ms/step                          |
| Total time                 | 1pc: 15 hours;  8pcs: 1.92 hours                          |
| Parameters (M)             | 177                                                         |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc> |

| Parameters                 | CNNCTC                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | GPU(Tesla V100-PCIE); CPU 2.60 GHz, 26 cores; Memory 790G; OS linux-gnu             |
| uploaded Date              | 07/06/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                      |
| Dataset                    |  MJSynth,SynthText                                                  |
| Training Parameters        | epoch=3,  batch_size=192          |
| Optimizer                  | RMSProp                                                         |
| Loss Function              | CTCLoss                                      |
| Speed                      | 1pc: 1180 ms/step;  8pcs: 1180 ms/step                          |
| Total time                 | 1pc: 62.9 hours;  8pcs: 8.67 hours                          |
| Parameters (M)             | 177                                                         |
| Scripts                    | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc> |

### Evaluation Performance

| Parameters          | CNNCTC                   |
| ------------------- | --------------------------- |
| Model Version       | V1                |
| Resource            |  Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 09/28/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | IIIT5K    |
| batch_size          | 192                         |
| outputs             | Accuracy                 |
| Accuracy            |  85%  |
| Model for inference | 675M (.ckpt file)         |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | CNNCTC                      |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 19/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | IIIT5K                      |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Accuracy=0.8642             |
| Model for inference | 675M(.ckpt file)            |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```python
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target=cfg.device_target)
  context.set_context(device_id=cfg.device_id)

  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```python
  # Load dataset
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False,                   loss_scale_manager=None)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
