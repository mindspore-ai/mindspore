# Contents

- [LSTM Description](#lstm-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Dataset Preparation](#dataset-preparation)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)


# [LSTM Description](#contents)

This example is for LSTM model training and evaluation.

[Paper](https://www.aclweb.org/anthology/P11-1015/):  Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, Christopher Potts. [Learning Word Vectors for Sentiment Analysis](https://www.aclweb.org/anthology/P11-1015/). Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. 2011

# [Model Architecture](#contents)

LSTM contains embeding, encoder and decoder modules. Encoder module consists of LSTM layer. Decoder module consists of fully-connection layer.


# [Dataset](#contents)

- aclImdb_v1 for training evaluation.[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- GloVe: Vector representations for words.[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)


# [Environment Requirements](#contents)

- Hardware（GPU/CPU）
- Framework
  - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/en/master/index.html)
  - [MindSpore API](https://www.mindspore.cn/api/en/master/index.html)


# [Quick Start](#contents)

- runing on GPU

  ```bash
  # run training example
  bash run_train_gpu.sh 0 ./aclimdb ./glove_dir

  # run evaluation example
  bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
  ```


# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
├── lstm
    ├── README.md               # descriptions about LSTM
    ├── script
    │   ├── run_eval_gpu.sh     # shell script for evaluation on GPU
    │   └── run_train_gpu.sh    # shell script for training on GPU
    ├── src
    │   ├── config.py           # parameter configuration
    │   ├── dataset.py          # dataset preprocess
    │   ├── imdb.py             # imdb dataset read script
    │   └── lstm.py             # Sentiment model
    ├── eval.py                 # evaluation script
    └── train.py                # training script
```


## [Script Parameters](#contents)

### Training Script Parameters

```python
usage: train.py  [-h] [--preprocess {true, false}] [--aclimdb_path ACLIMDB_PATH]
                 [--glove_path GLOVE_PATH] [--preprocess_path PREPROCESS_PATH]
                 [--ckpt_path CKPT_PATH] [--pre_trained PRE_TRAINING]
                 [--device_target {GPU, CPU}]

Mindspore LSTM Example

options:
  -h, --help                          # show this help message and exit
  --preprocess {true, false}          # whether to preprocess data.
  --aclimdb_path ACLIMDB_PATH         # path where the dataset is stored.
  --glove_path GLOVE_PATH             # path where the GloVe is stored.
  --preprocess_path PREPROCESS_PATH   # path where the pre-process data is stored.
  --ckpt_path CKPT_PATH               # the path to save the checkpoint file.
  --pre_trained                       # the pretrained checkpoint file path.
  --device_target                     # the target device to run, support "GPU", "CPU". Default: "GPU".
```


### Running Options

```python
config.py:
    num_classes                   # classes num
    learning_rate                 # value of learning rate
    momentum                      # value of momentum
    num_epochs                    # epoch size
    batch_size                    # batch size of input dataset
    embed_size                    # the size of each embedding vector
    num_hiddens                   # number of features of hidden layer
    num_layers                    # number of layers of stacked LSTM
    bidirectional                 # specifies whether it is a bidirectional LSTM
    save_checkpoint_steps         # steps for saving checkpoint files
```

### Network Parameters


## [Dataset Preparation](#contents)
- Download the dataset aclImdb_v1.

> Unzip the aclImdb_v1 dataset to any path you want and the folder structure should be as follows:
> ```
> .
> ├── train  # train dataset
> └── test   # infer dataset
> ```

- Download the GloVe file.

> Unzip the glove.6B.zip to any path you want and the folder structure should be as follows:
> ```
> .
> ├── glove.6B.100d.txt
> ├── glove.6B.200d.txt
> ├── glove.6B.300d.txt    # we will use this one later.
> └── glove.6B.50d.txt
> ```

> Adding a new line at the beginning of the file which named `glove.6B.300d.txt`.
> It means reading a total of 400,000 words, each represented by a 300-latitude word vector.
> ```
> 400000    300
> ```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate and network hyperparameters.

- Run `sh run_train_gpu.sh` for training.

    ``` bash
    bash run_train_gpu.sh 0 ./aclimdb ./glove_dir
    ```

    The above shell script will run distribute training in the background. You will get the loss value as following:
    ```shell
    # grep "loss is " log.txt
    epoch: 1 step: 390, loss is 0.6003723
    epcoh: 2 step: 390, loss is 0.35312173
    ...
    ```


## [Evaluation Process](#contents)

- Run `bash run_eval_gpu.sh` for evaluation.

    ``` bash
    bash run_eval_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt
    ```

# [Model Description](#contents)
## [Performance](#contents)

### Training Performance

| Parameters                 | LSTM                                                           |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | Tesla V100-SMX2-16GB                                           |
| uploaded Date              | 08/06/2020 (month/day/year)                                    |
| MindSpore Version          | 0.6.0-beta                                                     |
| Dataset                    | aclimdb_v1                                                     |
| Training Parameters        | epoch=20, batch_size=64                                        |
| Optimizer                  | Momentum                                                       |
| Loss Function              | Softmax Cross Entropy                                          |
| Speed                      | 1022 (1pcs)                                                    |
| Loss                       | 0.12                                                           |
| Params (M)                 | 6.45                                                           |
| Checkpoint for inference   | 292.9M (.ckpt file)                                            |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm |


### Evaluation Performance

| Parameters          | LSTM                        |
| ------------------- | --------------------------- |
| Resource            | Tesla V100-SMX2-16GB        |
| uploaded Date       | 08/06/2020 (month/day/year) |
| MindSpore Version   | 0.6.0-beta                  |
| Dataset             | aclimdb_v1                  |
| batch_size          | 64                          |
| Accuracy            | 84%                         |


# [Description of Random Situation](#contents)

There are three random situations:
- Shuffle of the dataset.
- Initialization of some model weights.


# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
