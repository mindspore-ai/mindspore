# DeepFM Description

This is an example of training DeepFM with Criteo dataset in MindSpore.

[Paper](https://arxiv.org/pdf/1703.04247.pdf) Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He


# Model architecture

The overall network architecture of DeepFM is show below:

[Link](https://arxiv.org/pdf/1703.04247.pdf)


# Requirements
- Install [MindSpore](https://www.mindspore.cn/install/en).
- Download the criteo dataset for pre-training. Extract and clean text in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). Convert the dataset to TFRecord format and move the files to a specified path.
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

# Script description

## Script and sample code

```python
├── deepfm       
  ├── README.md                      
  ├── scripts 
  │   ├──run_train.sh                  
  │   ├──run_eval.sh                    
  ├── src                              
  │   ├──config.py                     
  │   ├──dataset.py
  │   ├──callback.py                                    
  │   ├──deepfm.py
  ├── train.py
  ├── eval.py
```

## Training process

### Usage

- sh run_train.sh [DEVICE_NUM] [DATASET_PATH] [MINDSPORE_HCCL_CONFIG_PAHT]
- python train.py --dataset_path [DATASET_PATH]

### Launch

``` 
# distribute training example
  sh scripts/run_distribute_train.sh 8 /opt/dataset/criteo /opt/mindspore_hccl_file.json
# standalone training example
  sh scripts/run_standalone_train.sh 0 /opt/dataset/criteo
  or
  python train.py --dataset_path /opt/dataset/criteo > output.log 2>&1 &
```

### Result

Training result will be stored in the example path. 
Checkpoints will be stored at `./checkpoint` by default, 
and training log  will be redirected to `./output.log` by default,
and loss log will be redirected to `./loss.log` by default,
and eval log will be redirected to `./auc.log` by default. 


## Eval process

### Usage

- sh run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

``` 
# infer example
    sh scripts/run_eval.sh 0 ~/criteo/eval/ ~/train/deepfm-15_41257.ckpt
```

> checkpoint can be produced in training process. 

### Result

Inference result will be stored in the example path, you can find result like the followings in `auc.log`. 

``` 
2020-05-27 20:51:35 AUC: 0.80577889065281, eval time: 35.55999s.
```

# Model description

## Performance

### Training Performance

| Parameters                 | DeepFM                                                |
| -------------------------- | ------------------------------------------------------|
| Model Version              |                                                       |
| Resource                   | Ascend 910, cpu:2.60GHz 96cores, memory:1.5T          |
| uploaded Date              | 05/27/2020                                            |
| MindSpore Version          | 0.2.0                                                 |
| Dataset                    | Criteo                                                |
| Training Parameters        | src/config.py                                         |
| Optimizer                  | Adam                                                  |
| Loss Function              | SoftmaxCrossEntropyWithLogits                         |
| outputs                    |                                                       |
| Loss                       | 0.4234                                                |
| Accuracy                   | AUC[0.8055]                                           |
| Total time                 | 91 min                                                |
| Params (M)                 |                                                       |
| Checkpoint for Fine tuning |                                                       |
| Model for inference        |                                                       |

#### Inference Performance

| Parameters                 |                               |                           |
| -------------------------- | ----------------------------- | ------------------------- |
| Model Version              |                               |                           |   
| Resource                   | Ascend 910                    | Ascend 310                | 
| uploaded Date              | 05/27/2020                    | 05/27/2020                | 
| MindSpore Version          | 0.2.0                         | 0.2.0                     |  
| Dataset                    | Criteo                        |                           |
| batch_size                 | 1000                          |                           |
| outputs                    |                               |                           |
| Accuracy                   | AUC[0.8055]                   |                           |                      
| Speed                      |                               |                           |                     
| Total time                 | 35.559s                       |                           |                      
| Model for inference        |                               |                           |                 

# ModelZoo Homepage  
 [Link](https://gitee.com/mindspore/mindspore/tree/master/mindspore/model_zoo)  
