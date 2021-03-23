# Contents

- [TextCNN Description](#textcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TextCNN Description](#contents)

TextCNN is an algorithm that uses convolutional neural networks to classify text. It was proposed by Yoon Kim in the article "Convolutional Neural Networks for Sentence Classification" in 2014. It is widely used in various tasks of text classification (such as sentiment analysis). It has become the standard benchmark for the new text classification framework. Each module of TextCNN can complete text classification tasks independently, and it is convenient for distributed configuration and parallel execution. TextCNN is very suitable for the semantic analysis of short texts such as Weibo/News/E-commerce reviews and video bullet screens.

[Paper](https://arxiv.org/abs/1408.5882): Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.

# [Model Architecture](#contents)

The basic network structure design of TextCNN can refer to the paper "Convolutional Neural Networks for Sentence Classification". The specific implementation takes reading a sentence "I like this movie very much!" as an example. First, the word segmentation algorithm is used to divide the words into 7 words, and then the words in each part are expanded into a five-dimensional vector through the embedding method. Then use different convolution kernels ([3,4,5]*5) to perform convolution operations on them to obtain feature maps. The default number of convolution kernels is 2. Then use the maxpool operation to pool all the feature maps, and finally merge the pooling result into a one-dimensional feature vector through the connection operation. At last, it can be divided into 2 categories with softmax, and the positive/negative emotions are obtained.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [Movie Review Data](<http://www.cs.cornell.edu/people/pabo/movie-review-data/>)

- Dataset size：1.18M，5331 positive and 5331 negative processed sentences / snippets.
    - Train：1.06M, 9596 sentences / snippets
    - Test：0.12M, 1066 sentences / snippets
- Data format：text
    - Please click [here](<http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz>) to download the data, and change the files into utf-8. Then put it into the `data` directory.
    - Note：Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
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
  python train.py > train.log 2>&1 &
  OR
  sh scripts/run_train.sh

  # run evaluation example
  python eval.py > eval.log 2>&1 &
  OR
  sh scripts/run_eval.sh ckpt_path
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── textcnn
        ├── README.md                    // descriptions about textcnn
        ├──scripts
        │   ├── run_train.sh             // shell script for distributed on Ascend
        │   ├── run_eval.sh              // shell script for evaluation on Ascend
        ├── src
        │   ├── dataset.py             // Processing dataset
        │   ├── textcnn.py          // textcnn architecture
        │   ├── config.py            // parameter configuration
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── export.py               //  export checkpoint to other format file
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for movie review dataset

  ```python
  'pre_trained': 'False'    # whether training based on the pre-trained model
  'nump_classes': 2         # the number of classes in the dataset
  'batch_size': 64          # training batch size
  'epoch_size': 4           # total training epochs
  'weight_decay': 3e-5      # weight decay value
  'data_path': './data/'    # absolute full path to the train and evaluation datasets
  'device_target': 'Ascend' # device running the program
  'device_id': 0            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 1  # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_path': './train_textcnn.ckpt'  # the absolute full path to save the checkpoint file
  'word_len': 51            # The length of the word
  'vec_length': 40          # The length of the vector
  'base_lr': 1e-3          # The base learning rate
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

- running on Ascend

  ```python
  python train.py > train.log 2>&1 &
  OR
  sh scripts/run_train.sh
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files in `ckpt`. The loss value will be achieved as follows:

  ```python
  # grep "loss is " train.log
  epoch: 1 step 149, loss is 0.6194226145744324
  epoch: 2 step 149, loss is 0.38729554414749146
  ...
  ```

  The model checkpoint will be saved in the `ckpt` directory.

## [Evaluation Process](#contents)

- evaluation on movie review dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/textcnn/ckpt/train_textcnn.ckpt".

  ```python
  python eval.py --checkpoint_path=ckpt_path > eval.log 2>&1 &
  OR
  sh scripts/run_eval.sh ckpt_path
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.7971428571428572}
  ```

# [Model Description](#contents)

## [Performance](#contents)

### TextCNN on Movie Review Dataset

| Parameters          | Ascend                                                |
| ------------------- | ----------------------------------------------------- |
| Model Version       | TextCNN                                               |
| Resource            | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G      |
| uploaded Date       | 11/10/2020 (month/day/year)                           |
| MindSpore Version   | 1.0.1                                                 |
| Dataset             | Movie Review Data                                     |
| Training Parameters | epoch=4, steps=149, batch_size = 64                   |
| Optimizer           | Adam                                                  |
| Loss Function       | Softmax Cross Entropy                                 |
| outputs             | probability                                           |
| Loss                | 0.1724                                                |
| Speed               | 1pc: 12.069 ms/step                                   |
| Total time          | 1pc: 13s                                              |
| Scripts             | [textcnn script](https://gitee.com/xinyunfan/textcnn) |

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
