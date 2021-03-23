# Contents

- [Openpose Description](#googlenet-description)
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

# [Openpose Description](#contents)

Openpose network proposes a bottom-up human attitude estimation algorithm using Part Affinity Fields (PAFs). Instead of a top-down algorithm: Detect people first and then return key-points and skeleton. The advantage of openpose is that the computing time does not increase significantly as the number of people in the image increases.However,the top-down algorithm is based on the detection result, and the runtimes grow linearly with the number of people.

[Paper](https://arxiv.org/abs/1611.08050):  Zhe Cao,Tomas Simon,Shih-En Wei,Yaser Sheikh,"Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields",The IEEE Conference on Computer Vision and Pattern Recongnition(CVPR),2017

# [Model Architecture](#contents)

In first step the image is passed through baseline CNN network to extract the feature maps of the input In the paper. In this paper thee authors used first 10 layers of VGG-19 network.
The feature map is then process in a multi-stage CNN pipeline to generate the Part Confidence Maps and Part Affinity Field.
In the last step, the Confidence Maps and Part Affinity Fields  that are generated above are processed by a greedy bipartite matching algorithm to obtain the poses for each person in the image.

# [Dataset](#contents)

Prepare datasets, including training sets, verification sets, and annotations.The training set and validation set samples are located in the "dataset" directory, The available datasets include coco2014,coco2017 datasets.
In the currently provided training script, the coco2017 data set is used as an example to perform data preprocessing during the training process. If users use data sets in other formats, please modify the data set loading and preprocessing methods

- Download data from coco2017 data official website and unzip.

 ````bash
     wget http://images.cocodataset.org/zips/train2017.zip
     wget http://images.cocodataset.org/zips/val2017.zip
     wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
````

- Create the mask dataset.

    Run python gen_ignore_mask.py

````python
    python gen_ignore_mask.py --train_ann ../dataset/annotations/person_keypoints_train2017.json --val_ann ../dataset/annotations/person_keypoints_val2017.json --train_dir train2017 --val_dir val2017
````

- The dataset folder is generated in the root directory and contains the following files:

   ```python
   ├── dataset
       ├── annotation
           ├─person_keypoints_train2017.json
           └─person_keypoints_val2017.json
       ├─ignore_mask_train
       ├─ignore_mask_val
       ├─tran2017
       └─val2017
   ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware (Ascend)
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- Download the VGG19 model of the MindSpore version:
    - vgg19-0-97_5004.ckpt
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

  ```python
  # run training example
  python train.py --train_dir train2017 --train_ann person_keypoints_train2017.json > train.log 2>&1 &

  # run distributed training example
  bash run_distribute_train.sh [RANK_TABLE_FILE]

  # run evaluation example
  python eval.py --model_path path_to_eval_model.ckpt --imgpath_val ./dataset/val2017 --ann ./dataset/annotations/person_keypoints_val2017.json > eval.log 2>&1 &
  OR
  bash scripts/run_eval_ascend.sh
  ```

[RANK_TABLE_FILE] is the path of the multi-card information configuration table in the environment. The configuration table can be automatically generated by the tool [hccl_tool](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```python
├── ModelZoo_openpose_MS_MIT
    ├── README.md                        // descriptions about openpose
    ├── scripts
    │   ├──run_standalone_train.sh       // shell script for distributed on Ascend
    │   ├──run_distribute_train.sh       // shell script for distributed on Ascend with 8p
    │   ├──run_eval_ascend.sh            // shell script for evaluation on Ascend
    ├── src
    │   ├──openposenet.py                // Openpose architecture
    │   ├──loss.py                       // Loss function
    │   ├──config.py                     // parameter configuration
    │   ├──dataset.py                    // Data preprocessing
    │   ├──utils.py                      // Utils
    │   ├──gen_ignore_mask.py            // Generating mask data script
    ├── export.py                        // model conversion script
    ├── train.py                         // training script
    ├── eval.py                          // evaluation script  
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for openpose

  ```python
  'data_dir': 'path to dataset'                    # absolute full path to the train and evaluation datasets
  'vgg_path': 'path to vgg model'                  # absolute full path to vgg19 model
  'save_model_path': 'path of saving models'       # absolute full path to output models
  'load_pretrain': 'False'                         # whether training based on the pre-trained model
  'pretrained_model_path':''                       # load pre-trained model path
  'lr': 1e-4                                       # initial learning rate
  'batch_size': 10                                 # training batch size
  'lr_gamma': 0.1                                  # lr scale when reach lr_steps
  'lr_steps': '100000,200000,250000'               # the steps when lr * lr_gamma
  'loss scale': 16384                              # the loss scale of mixed precision
  'max_epoch_train': 60                            # total training epochs
  'insize': 368                                    # image size used as input to the model
  'keep_checkpoint_max': 1                         # only keep the last keep_checkpoint_max checkpoint
  'log_interval': 100                              # the interval of print a log
  'ckpt_interval': 5000                            # the interval of saving a output model
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py --train_dir train2017 --train_ann person_keypoints_train2017.json > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```python
  # grep "epoch " train.log
  epoch[0], iter[0], loss[0.29211228793809957], 0.13 imgs/sec, vgglr=0.0,baselr=2.499999936844688e-05,stagelr=9.999999747378752e-05
  epoch[0], iter[100], loss[0.060355084178521694], 24.92 imgs/sec, vgglr=0.0,baselr=2.499999936844688e-05,stagelr=9.999999747378752e-05
  epoch[0], iter[200], loss[0.026628130997662272], 26.20 imgs/sec, vgglr=0.0,baselr=2.499999936844688e-05,stagelr=9.999999747378752e-05
  ...
  ```

  The model checkpoint will be saved in the directory of config.py: 'save_model_path'.

## [Evaluation Process](#contents)

### Evaluation

- running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/openpose/outputs/\*time*\/0-6_30000.ckpt".

  ```python
  python eval.py --model_path path_to_eval_model.ckpt --imgpath_val ./dataset/val2017 --ann ./dataset/annotations/person_keypoints_val2017.json > eval.log 2>&1 &
  OR
  bash scripts/run_eval_ascend.sh
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "AP" eval.log

  {'AP': 0.40250956300341397, 'Ap .5': 0.6658941566481336, 'AP .75': 0.396047897339743, 'AP (M)': 0.3075356543635785, 'AP (L)': 0.533772768618845, 'AR': 0.4519836272040302, 'AR .5': 0.693639798488665, 'AR .75': 0.4570214105793451, 'AR (M)': 0.32155148866429945, 'AR (L)': 0.6330360460795242}

  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend
| -------------------------- | -----------------------------------------------------------
| Model Version              | openpose
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G
| uploaded Date              | 12/14/2020 (month/day/year)
| MindSpore Version          | 1.0.1-alpha
| Training Parameters        | epoch=60(1pcs)/80(8pcs), steps=30k(1pcs)/5k(8pcs), batch_size=10, init_lr=0.0001
| Optimizer                  | Adam(1pcs)/Momentum(8pcs)
| Loss Function              | MSE
| outputs                    | pose
| Speed                      | 1pcs: 35fps, 8pcs: 230fps
| Total time                 | 1pcs: 22.5h, 8pcs: 5.1h
| Checkpoint for Fine tuning | 602.33M (.ckpt file)
