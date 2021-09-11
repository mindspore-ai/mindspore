# Contents

[查看中文](./README_CN.md)

- [Vit Description](#vit-description)
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
    - [Export Process](#Export-process)
        - [Export](#Export)
    - [Inference Process](#Inference-process)
        - [Inference](#Inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Vit Description](#contents)

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

[Paper](https://arxiv.org/abs/2010.11929):  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.

# [Model Architecture](#contents)

Specifically, the vit contains transformer encoder. The structure is patch_embeding + n transformer layer + head(FC for classification).

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```bash
└─dataset
    ├─train                # train dataset, should be .tar file when running on clould
    └─val                  # evaluate dataset
```

- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # run training example CONFIG_PATH in ./config/*.yml or *.ymal
  python train.py --config_path=[CONFIG_PATH] > train.log 2>&1 &

  # run distributed training example
  cd scripts;
  bash run_train_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # run evaluation example
  cd scripts;
  bash run_eval.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # run inferenct example
  cd scripts;
  bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [DEVICE_ID]
  ```

  For distributed training, a hccl configuration file(RANK_TABLE_FILE) with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train imagenet 8p on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/config/vit_patch32_imagenet2012_config_cloud.yml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=1" on yml file.
      #          Set "output_path" on yml file.
      #          Set "data_path='/cache/data/ImageNet/'" on yml file.
      #          Set other parameters on yml file you need.
      #       b. Add "enable_modelarts=1" on the website UI interface.
      #          Set "output_path" on yml file.
      #          Set "data_path='/cache/data/ImageNet/'" on yml file.
      #          Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/vit" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Eval imagenet on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/config/vit_eval.yml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=1" on yml file.
      #          Set "output_path" on yml file.
      #          Set "data_path='/cache/data/ImageNet/'" on yml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on yml file.
      #          Set "load_path='/cache/checkpoint_path/model.ckpt'" on yml file.
      #          Set other parameters on yml file you need.
      #       b. Add "enable_modelarts=1" on the website UI interface.
      #          Add "dataset_name=imagenet" on the website UI interface.
      #          Add "val_data_path=/cache/data/ImageNet/val/" on the website UI interface.
      #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
      #          Add "load_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your pretrained model to S3 bucket.
      # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (5) Set the code directory to "/path/vit" on the website UI interface.
      # (6) Set the startup file to "eval.py" on the website UI interface.
      # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (8) Create your job.
      ```

    - Export on ModelArts

      ```python
      # (1) Add "config_path='/path_to_code/config/vit_export.yml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=1" on yml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on yml file.
      #          Set "load_path='/cache/checkpoint_path/model.ckpt'" on yml file.
      #          Set other parameters on yml file you need.
      #       b. Add "enable_modelarts=1" on the website UI interface.
      #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
      #          Add "load_path=/cache/checkpoint_path/model.ckpt" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your trained model to S3 bucket.
      # (4) Set the code directory to "/path/vit" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                            // descriptions about all the models
    ├── vit
        ├── README.md                        // descriptions about vit
        ├── ascend310_infer                  // application for 310 inference
        ├── scripts
        │   ├──run_train_distribute.sh       // shell script for distributed on Ascend
        │   ├──run_train_standalone.sh       // shell script for single node on Ascend
        │   ├──run_eval.sh                   // shell script for evaluation on Ascend
        │   ├──run_infer_310.sh              // shell script for 310 inference
        ├── src
        │   ├──autoaugment.py                // autoaugment for data processing
        │   ├──callback.py                   // logging callback
        │   ├──cross_entropy.py              // ce loss
        │   ├──dataset.py                    // creating dataset
        │   ├──eval_engine.py                // eval code
        │   ├──logging.py                    // logging engine
        │   ├──lr_generator.py               // lr schedule
        │   ├──metric.py                     // metric for eval
        │   ├──optimizer.py                  // user defined optimizer
        │   ├──vit.py                        // model architecture
        │   ├──model_utils                   // cloud depending files, all model zoo shares the same files, not recommend user changing
        ├── config
        │   ├──vit_eval.yml                                    // parameter configuration for eval
        │   ├──vit_export.yml                                  // parameter configuration for export
        │   ├──vit_patch32_imagenet2012_config.yml             // parameter configuration for 8P training
        │   ├──vit_patch32_imagenet2012_config_cloud.yml       // parameter configuration for 8P training on cloud
        │   ├──vit_patch32_imagenet2012_config_standalone.yml  // parameter configuration for 1P training
        ├── train.py                         // training script
        ├── eval.py                          //  evaluation script
        ├── postprogress.py                  // post process for 310 inference
        ├── export.py                        // export checkpoint files into air/mindir
        ├── create_imagenet2012_label.py     // create label for 310 inference
        ├── requirements.txt                 // requirements pip list
        ├── mindspore_hub_conf.py            // The mindspore_hub_conf file required for the operation of the hub warehouse
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for vit, ImageNet dataset

  ```python
  enable_modelarts: 1               # train on cloud or not

  # Url for modelarts
  data_url: ""                      # S3 dataset path
  train_url: ""                     # S3 output path
  checkpoint_url: ""                # S3 pretrain model path
  output_path: "/cache/train"       # output cache, copy to train_url
  data_path: "/cache/datasets/imagenet" # dataset cache(real path on cloud), copy from data_url
  load_path: "/cache/model/vit_base_patch32.ckpt" # model cache, copy from checkpoint_url

  # train datasets
  dataset_path: '/cache/datasets/imagenet/train' # training dataset
  train_image_size: 224             # image height and weight used as input to the model
  interpolation: 'BILINEAR'         # dataset interpolation
  crop_min: 0.05                    # random crop min value
  batch_size: 256                   # batch size for train
  train_num_workers: 14             # parallel work number

  # eval datasets
  eval_path: '/cache/datasets/imagenet/val' # eval dataset
  eval_image_size: 224              # image height and weight used as input to the model
  eval_batch_size: 256              # batch size for eval
  eval_interval: 1                  # eval interval
  eval_offset: -1                   # eval offset
  eval_num_workers: 12              # parallel work number

  # network
  backbone: 'vit_base_patch32'      # backbone type
  class_num: 1001                   # class number, imagenet is 1000+1
  vit_config_path: 'src.vit.VitConfig' #vit config path, for advanced user to design transformer based new architecture
  pretrained: ''                    # pre-trained model path, '' means not use pre-trained model

  # lr
  lr_decay_mode: 'cosine'           # lr decay type, support cosine, exp... detail see lr_generator.py
  lr_init: 0.0                      # start lr(epoch 0)
  lr_max: 0.00355                   # max lr
  lr_min: 0.0                       # min lr (max epoch)
  max_epoch: 300                    # max epoch
  warmup_epochs: 40                 # warmup epoch

  # optimizer
  opt: 'adamw'                      # optimizer type
  beta1: 0.9                        # adam beta
  beta2: 0.999                      # adam beta
  weight_decay: 0.05                # weight decay
  no_weight_decay_filter: "beta,bias" # which type of weight not use weight decay
  gc_flag: 0                        # use gc or not, not support for user defined opt, support for system defined opt

  # loss, some parameter also used in datasets
  loss_scale: 1024                  # amp loss scale
  use_label_smooth: 1               # use label smooth or not
  label_smooth_factor: 0.1          #label smooth factor
  mixup: 0.2                        # use mixup or not
  autoaugment: 1                    # use autoaugment or not
  loss_name: "ce_smooth_mixup"      #loss type, detail see cross_entropy.py

  # ckpt
  save_checkpoint: 1                # save .ckpt(training result) or not
  save_checkpoint_epochs: 8         # when to save .ckpt
  keep_checkpoint_max: 3            # max keep ckpt
  save_checkpoint_path: './outputs' # save path

  # profiler
  open_profiler: 0 # do profiling or not. if use profile, you'd better set a small dataset as training dataset and set max_epoch=1
  ```

For more configuration details, please refer the script `train.py`, `eval.py`, `export.py` and `config/*.yml`.

## [Training Process](#contents)

### Training

- running on Ascend

  ```bash
  python train.py --config_path=[CONFIG_PATH] > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash
  # vim log
  2021-08-05 15:17:12:INFO:compile time used=143.16s
  2021-08-05 15:34:41:INFO:epoch[0], epoch time: 1048.72s, per step time: 0.2096s, loss=6.738676, lr=0.000011, fps=1221.51
  2021-08-05 15:52:03:INFO:epoch[1], epoch time: 1041.90s, per step time: 0.2082s, loss=6.381927, lr=0.000022, fps=1229.51
  ...
  ```

  The model checkpoint will be saved in the train directory.

### Distributed Training

- running on Ascend

  ```bash
  cd scripts
  bash run_train_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log`. The loss value will be achieved as follows:

  ```bash
  # vim train_parallel0/log
  # fps depend on cpu processing ability, data processing take times
  2021-08-05 20:15:16:INFO:compile time used=191.77s
  2021-08-05 20:17:46:INFO:epoch[0], epoch time: 149.10s, per step time: 0.2386s, loss=6.729037, lr=0.000089, fps=8584.97, accuracy=0.014940, eval_cost=1.58
  2021-08-05 20:20:11:INFO:epoch[1], epoch time: 143.44s, per step time: 0.2295s, loss=6.786729, lr=0.000177, fps=8923.72, accuracy=0.047000, eval_cost=1.27

  ...
  2021-08-06 08:18:18:INFO:epoch[299], epoch time: 143.19s, per step time: 0.2291s, loss=2.718115, lr=0.000000, fps=8939.29, accuracy=0.741800, eval_cost=1.28
  2021-08-06 08:18:20:INFO:training time used=43384.70s
  2021-08-06 08:18:20:INFO:last_metric[0.74206]
  2021-08-06 08:18:20:INFO:ip[*.*.*.*], mean_fps[8930.40]

  ```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on imagenet dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/vit/vit_base_patch32.ckpt".

  ```bash
  cd scripts
  bash run_eval.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.741260
  ```

  Note that for evaluation after distributed training, please choose the checkpoint_path to be the saved checkpoint file such as "username/vit/train_parallel0/outputs/vit_base_patch32-288_625.ckpt". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.741260
  ```

## [Export Process](#contents)

### [Export](#content)

Before export model, you must modify the config file, config/export.yml.
The config items you should modify are batch_size and ckpt_file/pretrained.

```bash
python export.py --config_path=[CONFIG_PATH]
```

## [Inference Process](#contents)

### [Inference](#content)

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.
Current batch_ Size can only be set to 1.

- inference on imagenet dataset when running on Ascend

  Before running the command below, you should modify the config file. The items you should modify are batch_size and val_data_path.

  Inference result will be stored in the example path, you can find result like the followings in acc.log.

  ```shell
  # Ascend310 inference
  cd scripts
  bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.74084, top5 accuracy: 0.91026
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### Vit on imagenet 1200k images

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | Vit                                                         |
| Resource                   | Ascend 910; CPU 2.60GHz, 56cores; Memory 314G; OS Euler2.8  |
| uploaded Date              | 08/30/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | 1200k images                                                |
| Training Parameters        | epoch=300, steps=625*300, batch_size=256, lr=0.00355        |
| Optimizer                  | Adamw                                                       |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.0                                                         |
| Speed                      | 1pc: 180 ms/step;  8pcs: 185 ms/step                        |
| Total time                 | 8pcs: 11 hours                                              |
| Parameters (M)             | 86.0                                                        |
| Checkpoint for Fine tuning | 1000M (.ckpt file)                                          |
| Scripts                    | [vit script](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/vit) |

### Inference Performance

#### Vit on 1200k images

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Vit                         |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 08/30/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | 1200k images                |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | 8pcs: 73.5%-74.6%           |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```python
  # get args from cfg and get parameter by args
  args.loss_scale = ...
  lrs = ...
  ...
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target=args.device_target)
  context.set_context(device_id=args.device_id)

  # Load unseen dataset for inference
  dataset = dataset.create_dataset(args.data_path, 1, False)

  # Define model
  net = ViT(args.vit_config)
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(args.pretrained)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```python
  # get args from cfg and get parameter by args
  args.loss_scale = ...
  lrs = ...
  ...

  # Load dataset
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = ViT(args.vit_config)
  # Continue training if set pre_trained to be True
  if cfg.pretrained != '':
      param_dict = load_checkpoint(cfg.pretrained)
      load_param_into_net(net, param_dict)
  # Define model
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Start training
  epoch_size = args.max_epoch
  step_size = dataset.get_dataset_size()
  # Set callbacks
  state_cb = StateMonitor(data_size=step_size,
                          tot_batch_size=args.batch_size * device_num,
                          lrs=lrs,
                          eval_interval=args.eval_interval,
                          eval_offset=args.eval_offset,
                          eval_engine=eval_engine,
                          logger=args.logger.info)
  cb = [state_cb, ]
  model.train(epoch_size, dataset, callbacks=cb, sink_size=step_size)
  print("train success")
  ```

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
