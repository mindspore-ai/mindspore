# MobileNetV2 Description


MobileNetV2 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1801.04381) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for MobileNetV2." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

# Model architecture

The overall network architecture of MobileNetV2 is show below:

[Link](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)

# Dataset

Dataset used: imagenet

- Dataset size: ~125G, 1.2W colorful images in 1000 classes
	- Train: 120G, 1.2W images
	- Test: 5G, 50000 images
- Data format: RGB images.
	- Note: Data will be processed in src/dataset.py 


# Features


# Environment Requirements

- Hardware（Ascend/GPU）
  - Prepare hardware environment with Ascend or GPU processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)


# Script description

## Script and sample code

```python
├── MobileNetV2        
  ├── Readme.md                      
  ├── scripts 
  │   ├──run_train.sh                  
  │   ├──run_eval.sh                    
  ├── src                              
  │   ├──config.py                     
  │   ├──dataset.py
  │   ├──luanch.py       
  │   ├──lr_generator.py                                 
  │   ├──mobilenetV2.py
  ├── train.py
  ├── eval.py
```

## Training process

### Usage

- Ascend: sh run_train.sh Ascend [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH]
- GPU: sh run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH]

### Launch

``` 
# training example
  Ascend: sh run_train.sh Ascend 8 192.168.0.1 0,1,2,3,4,5,6,7 ~/imagenet/train/
  GPU: sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 ~/imagenet/train/
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train/train.log` like followings. 

``` 
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## Eval process

### Usage

- Ascend: sh run_infer.sh Ascend [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: sh run_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

``` 
# infer example
    Ascend: sh run_infer.sh Ascend ~/imagenet/val/ ~/train/mobilenet-200_625.ckpt
    GPU: sh run_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-200_625.ckpt
```

> checkpoint can be produced in training process. 

### Result

Inference result will be stored in the example path, you can find result like the followings in `val.log`. 

``` 
result: {'acc': 0.71976314102564111} ckpt=/path/to/checkpoint/mobilenet-200_625.ckpt
```

# Model description

## Performance

### Training Performance

| Parameters                 | MobilenetV2                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              |                                                            | large                     |
| Resource                   | Ascend 910, cpu:2.60GHz 56cores, memory:314G               | NV SMX2 V100-32G          |
| uploaded Date              | 05/06/2020                                                 | 05/06/2020                |
| MindSpore Version          | 0.3.0                                                      | 0.3.0                     |
| Dataset                    | ImageNet                                                   | ImageNet                  |
| Training Parameters        | src/config.py                                              | src/config.py             |
| Optimizer                  | Momentum                                                   | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    |                                                            |                           |
| Loss                       |                                                            | 1.913                     |
| Accuracy                   |                                                            | ACC1[77.09%] ACC5[92.57%] |
| Total time                 |                                                            |                           |
| Params (M)                 |                                                            |                           |
| Checkpoint for Fine tuning |                                                            |                           |
| Model for inference        |                                                            |                           |

#### Inference Performance

| Parameters                 |                               |                           |                      |
| -------------------------- | ----------------------------- | ------------------------- | -------------------- |
| Model Version              | V1                            |                           |                      |
| Resource                   | Huawei 910                    | NV SMX2 V100-32G          | Huawei 310           |
| uploaded Date              | 05/06/2020                    | 05/22/2020                |                      |
| MindSpore Version          | 0.2.0                         | 0.2.0                     | 0.2.0                | 
| Dataset                    | ImageNet, 1.2W                | ImageNet, 1.2W            | ImageNet, 1.2W       |
| batch_size                 |                               | 130(8P)                   |                      |
| outputs                    |                               |                           |                      |
| Accuracy                   |                               | ACC1[72.07%] ACC5[90.90%] |                      |
| Speed                      |                               |                           |                      |
| Total time                 |                               |                           |                      |
| Model for inference        |                               |                           |                      |

# ModelZoo Homepage  
 [Link](https://gitee.com/mindspore/mindspore/tree/master/mindspore/model_zoo)  