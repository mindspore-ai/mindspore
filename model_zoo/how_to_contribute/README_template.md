<TOC>

# Title, Model name

> The Description of Model. The paper present this model.

## Model Architecture

> There could be various architecture about some model. Represent the architecture of your implementation.

## Features(optional)

> Represent the distinctive feature you used in the model implementation. Such as distributed auto-parallel or some special training trick.

## Dataset

> Provide the information of the dataset you used. Check the copyrights of the dataset you used, usually don't provide the hyperlink to download the dataset.

## Requirements

> Provide details of the software required, including:
>
> * The additional python package required. Add a `requirements.txt` file to the root dir of model for installing dependencies.
> * The necessary third-party code.
> * Some other system dependencies.
> * Some additional operations before training or prediction.

## Quick Start

> How to take a try without understanding anything about the model.

## Script Description

> The section provide the detail of implementation.

### Scripts and Sample Code

> Explain every file in your project.

### Script Parameter

> Explain every parameter of the model. Especially the parameters in `config.py`.

## Training

> Provide training information.

### Training Process

> Provide the usage of training scripts.

e.g. Run the following command for distributed training on Ascend.

```shell
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

### Transfer Training(Optional)

> Provide the guidelines about how to run transfer training based on an pretrained model.

### Training Result

> Provide the result of training.

e.g. Training checkpoint will be stored in `XXXX/ckpt_0`. You will get result from log file like the following:

```
epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
```

## Evaluation

### Evaluation Process

> Provide the use of evaluation scripts.

### Evaluation Result

> Provide the result of evaluation.

## Performance

### Training Performance

> Provide the detail of training performance including finishing loss, throughput, checkpoint size and so on.

### Inference Performance

> Provide the detail of evaluation performance including latency, accuracy and so on.

## Description of Random Situation

> Explain the random situation in the project.

## ModeZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).