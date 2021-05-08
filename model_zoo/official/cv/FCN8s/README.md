# Contents

- [FCN Description](#fcn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Data Generation](#data-generation)
        - [Training Data](#training-data)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FCN Description](#contents)

FCN is mainly used in the field of image segmentation, which is an end-to-end segmentation method. FCN changes the last full connected layers of VGG to process images of any size, reduce the parameters and improve the segmentation speed of the model. FCN uses VGG structure in the encoding part and deconvolution / up sampling operation in the decoding part to recover the image resolution. Finally, FCN8s uses 8 times deconvolution / up sampling operation to restore the output image to the same size as the input image.

[Paper]: Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

# [Model Architecture](#contents)

FCN8s uses VGG16 without the full connected layers as the encoding part, and fuses the features of the 3rd, 4th and 5th pooling layers in VGG16 respectively. Finally, the deconvolution of stride 8 is used to obtain the segmented image.

# [Dataset](#contents)

Dataset used:

[PASCAL VOC 2012](<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>)

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Architecture
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore through the official website, you can start training and evaluation by following these steps：

- running on Ascend with default parameters

  ```bash
  # run training example
  python train.py --device_id device_id

  # run evaluation example with default parameters
  python eval.py --device_id device_id
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── cv
    ├── FCN8s
        ├── README.md                 // descriptions about FCN
        ├── scripts
            ├── run_train.sh
            ├── run_standalone_train.sh
            ├── run_eval.sh
            ├── build_data.sh
        ├── src
        │   ├──data
        │       ├──build_seg_data.py       // creating dataset
        │       ├──dataset.py          // loading dataset
        │   ├──nets
        │       ├──FCN8s.py            // FCN-8s architecture
        │   ├──loss
        │       ├──loss.py            // loss function
        │   ├──utils
        │       ├──lr_scheduler.py            // getting learning_rateFCN-8s  
        ├── train.py                 // training script
        ├── eval.py                  //  evaluation script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for FCN8s

  ```python
     # dataset
    'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord', # path and name of one mindrecord file
    'batch_size': 32,
    'crop_size': 512,
    'image_mean': [103.53, 116.28, 123.675],
    'image_std': [57.375, 57.120, 58.395],
    'min_scale': 0.5,
    'max_scale': 2.0,
    'ignore_label': 255,
    'num_classes': 21,

    # optimizer
    'train_epochs': 500,
    'base_lr': 0.015,
    'loss_scale': 1024.0,

    # model
    'model': 'FCN8s',
    'ckpt_vgg16': '',
    'ckpt_pre_trained': '',

    # train
    'save_steps': 330,
    'keep_checkpoint_max': 5,
    'ckpt_dir': './ckpt',
  ```

For more information, see `config.py`.

## [Data Generation](#contents)

### Training Data

- build mindrecord training data

  ```bash
  sh build_data.sh
  or
  python src/data/build_seg_data.py  --data_root=/home/sun/data/Mindspore/benchmark_RELEASE/dataset  \
                                     --data_lst=/home/sun/data/Mindspore/benchmark_RELEASE/dataset/trainaug.txt  \
                                     --dst_path=dataset/MINDRECORED_NAME.mindrecord  \
                                     --num_shards=1  \
                                     --shuffle=True
  ```

## [Training Process](#contents)

### Training

- running on Ascend with default parameters

  ```bash
  python train.py --device_id device_id
  ```

  Checkpoints will be stored in the default path

## [Evaluation Process](#contents)

### Evaluation

- Evaluated on Pascal VOC 2012 validation set using Ascend

  Before running the command, check the path of the checkpoint used for evaluation. Please set the absolute path of the checkpoint

  ```bash
  python eval.py
  ```

  After running the above command, you can see the evaluation results on the terminal. The accuracy on the test set is presented as follows:

  ```bash
  mean IoU  0.6425
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend
| -------------------------- | -----------------------------------------------------------
| Model Version              | FCN-8s
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8
| uploaded Date              | 12/30/2020 (month/day/year)
| MindSpore Version          | 1.1.0-alpha
| Dataset                    | PASCAL VOC 2012
| Training Parameters        | epoch=500, steps=330, batch_size = 32, lr=0.015
| Optimizer                  | Momentum
| Loss Function              | Softmax Cross Entropy
| outputs                    | probability
| Loss                       | 0.038
| Speed                      | 1pc: 564.652 ms/step;
| Scripts                    | [FCN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/FCN8s)

### Inference Performance

#### FCN8s on PASCAL VOC

| Parameters          | Ascend
| ------------------- | ---------------------------
| Model Version       | FCN-8s
| Resource            | Ascend 910; OS Euler2.8
| Uploaded Date       | 12/30/2020 (month/day/year)
| MindSpore Version   | 1.1.0-alpha
| Dataset             | PASCAL VOC 2012
| batch_size          | 16
| outputs             | probability
| mean IoU            | 0.627

# [Description of Random Situation](#contents)

We set the random seeds in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

