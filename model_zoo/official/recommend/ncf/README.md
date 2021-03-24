# Contents

- [NCF Description](#NCF-description)
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
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NCF Description](#contents)

NCF is a general framework for collaborative filtering of recommendations in which a neural network architecture is used to model user-item interactions. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

[Paper](https://arxiv.org/abs/1708.05031):  He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.

# [Model Architecture](#contents)

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer. [neumf_model.py](neumf_model.py) defines the architecture details.

# [Dataset](#contents)

The [MovieLens datasets](http://files.grouplens.org/datasets/movielens/) are used for model training and evaluation. Specifically, we use two datasets: **ml-1m** (short for MovieLens 1 million) and **ml-20m** (short for MovieLens 20 million).

## ml-1m

ml-1m dataset contains 1,000,209 anonymous ratings of approximately 3,706 movies made by 6,040 users who joined MovieLens in 2000. All ratings are contained in the file "ratings.dat" without header row, and are in the following format:

```cpp
  UserID::MovieID::Rating::Timestamp
```

- UserIDs range between 1 and 6040.
- MovieIDs range between 1 and 3952.
- Ratings are made on a 5-star scale (whole-star ratings only).

## ml-20m

ml-20m dataset contains 20,000,263 ratings of 26,744 movies by 138493 users. All ratings are contained in the file "ratings.csv". Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

```text
userId,movieId,rating,timestamp
```

- The lines within this file are ordered first by userId, then, within user, by movieId.
- Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

In both datasets, the timestamp is represented in seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970. Each user has at least 20 ratings.

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
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
#run data process
bash scripts/run_download_dataset.sh

# run training example
bash scripts/run_train.sh

# run distributed training example
sh scripts/run_train.sh rank_table.json

# run evaluation example
sh run_eval.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── ModelZoo_NCF_ME
    ├── README.md                          // descriptions about NCF
    ├── scripts
    │   ├──run_train.sh                    // shell script for train
    │   ├──run_distribute_train.sh         // shell script for distribute train
    │   ├──run_eval.sh                     // shell script for evaluation
    │   ├──run_download_dataset.sh         // shell script for dataget and process
    │   ├──run_transfer_ckpt_to_air.sh     // shell script for transfer model style
    ├── src
    │   ├──dataset.py                      // creating dataset
    │   ├──ncf.py                          // ncf architecture
    │   ├──config.py                       // parameter configuration
    │   ├──movielens.py                    // data download file
    │   ├──callbacks.py                    // model loss and eval callback file
    │   ├──constants.py                    // the constants of model
    │   ├──export.py                       // export checkpoint files into geir/onnx
    │   ├──metrics.py                      // the file for auc compute
    │   ├──stat_utils.py                   // the file for data process functions
    ├── train.py               // training script
    ├── eval.py               //  evaluation script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for NCF, ml-1m dataset

  ```python
  * `--data_path`: This should be set to the same directory given to the data_download data_dir argument.
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is ml-1m.
  * `--train_epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--num_neg`: The Number of negative instances to pair with a positive instance.
  * `--layers`： The sizes of hidden layers for MLP.
  * `--num_factors`：The Embedding size of MF model.
  * `--output_path`：The location of the output file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.
  ```

## [Training Process](#contents)

### Training

  ```python
  bash scripts/run_train.sh
  ```

  The python command above will run in the background, you can view the results through the file `train.log`. After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```python
  # grep "loss is " train.log
  ds_train.size: 95
  epoch: 1 step: 95, loss is 0.25074288
  epoch: 2 step: 95, loss is 0.23324402
  epoch: 3 step: 95, loss is 0.18286772
  ...  
  ```

  The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ml-1m dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "checkpoint/ncf-125_390.ckpt".

  ```python
  sh scripts/run_eval.sh
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  HR:0.6846,NDCG:0.410
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | NCF                                                 |
| Resource                   | Ascend 910 ；CPU 2.60GHz，56cores；Memory，314G               |
| uploaded Date              | 10/23/2020 (month/day/year)                                  |
| MindSpore Version          | 1.0.0                                                |
| Dataset                    | ml-1m                                                        |
| Training Parameters        | epoch=25, steps=19418, batch_size = 256, lr=0.00382059       |
| Optimizer                  | GradOperation                                                |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probability                                                  |
| Speed                      | 1pc: 0.575 ms/step                                          |
| Total time                 | 1pc: 5 mins                       |

### Inference Performance

| Parameters          | Ascend              |
| ------------------- | --------------------------- |
| Model Version       | NCF               |
| Resource            | Ascend 910                  |
| Uploaded Date       | 10/23/2020 (month/day/year)  |
| MindSpore Version   | 1.0.0                |  
| Dataset             | ml-1m                       |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | HR:0.6846,NDCG:0.410        |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html). Following the steps below, this is a simple example:

<https://www.mindspore.cn/tutorial/inference/en/master/multi_platform_inference.html>

  ```python
  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
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

  ```python
  # Load dataset
  dataset = create_dataset(cfg.data_path, cfg.epoch_size)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

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

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
