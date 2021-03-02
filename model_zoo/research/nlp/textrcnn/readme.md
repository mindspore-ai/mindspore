# TextRCNN

## Contents

- [TextRCNN Description](#textrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [ModelZoo Homepage](#modelzoo-homepage)

## [TextRCNN Description](#contents)

TextRCNN, a model for text classification, which is proposed by the Chinese Academy of Sciences in 2015.
TextRCNN actually combines RNN and CNN, first uses bidirectional RNN to obtain upper semantic and grammatical information of the input text,
and then uses maximum pooling to automatically filter out the most important feature.
Then connect a fully connected layer for classification.

The TextCNN network structure contains a convolutional layer and a pooling layer. In RCNN, the feature extraction function of the convolutional layer is replaced by RNN. The overall structure consists of  RNN and pooling layer, so it is called RCNN.

[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552):  Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao: Recurrent Convolutional Neural Networks for Text Classification. AAAI 2015: 2267-2273

## [Model Architecture](#contents)

Specifically, the TextRCNN is mainly composed of three parts: a recurrent structure layer, a max-pooling layer, and a fully connected layer. In the paper, the length of the word vector $|e|=50$, the length of the context vector $|c|=50$, the hidden layer size $ H=100$, the learning rate $\alpha=0.01$, the amount of words is $|V|$, the input is a sequence of words, and the output is a vector containing categories.

## [Dataset](#contents)

Dataset used: [Sentence polarity dataset v1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

- Dataset size：10662 movie comments in 2 classes, 9596 comments for train set, 1066 comments for test set.
- Data format：text files. The processed data is in ```./data/```

## [Environment Requirements](#contents)

- Hardware: Ascend
- Framework: [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：[MindSpore tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html), [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html).

## [Quick Start](#contents)

- Preparing environment

```python
  # download the pretrained GoogleNews-vectors-negative300.bin, put it into /tmp
  # you can download from https://code.google.com/archive/p/word2vec/,
  # or from https://pan.baidu.com/s/1NC2ekA_bJ0uSL7BF3SjhIg, code: yk9a

  mv /tmp/GoogleNews-vectors-negative300.bin ./word2vec/
```

- Preparing data

```python
  # split the dataset by the following scripts.
  mkdir -p data/test && mkdir -p data/train
  python data_helpers.py --task dataset_split --data_dir dataset_dir

```

- Running on Ascend

```python
# run training
DEVICE_ID=7 python train.py
# or you can use the shell script to train in background
bash scripts/run_train.sh

# run evaluating
DEVICE_ID=7 python eval.py --ckpt_path {checkpoint path}
# or you can use the shell script to evaluate in background
bash scripts/run_eval.sh
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```python
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── textrcnn
        ├── README.md                    // descriptions about TextRCNN
        ├── data_src
        │   ├──rt-polaritydata            // directory to save the source data
        │   ├──rt-polaritydata.README.1.0.txt    // readme file of dataset
        ├── scripts
        │   ├──run_train.sh             // shell script for train on Ascend
        │   ├──run_eval.sh              // shell script for evaluation on Ascend
        │   ├──sample.txt              // example shell to run the above the two scripts
        ├── src
        │   ├──dataset.py             // creating dataset
        │   ├──textrcnn.py          // textrcnn architecture
        │   ├──config.py            // parameter configuration
        ├── train.py               // training script
        ├── export.py             // export script
        ├── eval.py               //  evaluation script
        ├── data_helpers.py               //  dataset split script
        ├── sample.txt               //  the shell to train and eval the model without scripts
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Textrcnn, Sentence polarity dataset v1.0.

  ```python
  'num_epochs': 10, # total training epochs
  'lstm_num_epochs': 15, # total training epochs when using lstm
  'batch_size': 64, # training batch size
  'cell': 'gru', # the RNN architecture, can be 'vanilla', 'gru' and 'lstm'.
  'ckpt_folder_path': './ckpt', # the path to save the checkpoints
  'preprocess_path': './preprocess', # the directory to save the processed data
  'preprocess' : 'false', # whethere to preprocess the data
  'data_path': './data/', # the path to store the splited data
  'lr': 1e-3, # the training learning rate
  'lstm_lr_init': 2e-3, # learning rate initial value when using lstm
  'lstm_lr_end': 5e-4, # learning rate end value when using lstm
  'lstm_lr_max': 3e-3, # learning eate max value when using lstm
  'lstm_lr_warm_up_epochs': 2 # warm up epoch num when using lstm
  'lstm_lr_adjust_epochs': 9 # lr adjust in lr_adjust_epoch, after that, the lr is lr_end when using lstm
  'emb_path': './word2vec', # the directory to save the embedding file
  'embed_size': 300, # the dimension of the word embedding
  'save_checkpoint_steps': 149, # per step to save the checkpoint
  'keep_checkpoint_max': 10 # max checkpoints to save
  ```

### Performance

| Model                 | MindSpore + Ascend                        | TensorFlow+GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| Resource                   | Ascend 910                    | NV SMX2 V100-32G          |
| Version          | 1.0.1                         | 1.4.0                     |
| Dataset                    | Sentence polarity dataset v1.0                    | Sentence polarity dataset v1.0            |
| batch_size                 | 64                        | 64                   |
| Accuracy                   | 0.78                      | 0.78 |
| Speed                      | 35ms/step                  |  77ms/step                         |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
