# Release 0.3.0-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * DeepFM: a factorization-machine based neural network for CTR prediction on Criteo dataset.
    * DeepLabV3: significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2007 semantic image segmentation benchmark.
    * Faster-RCNN: towards real-time object detection with region proposal networks on COCO 2017 dataset.
    * SSD: a single stage object detection methods on COCO 2017 dataset.
    * GoogLeNet: a deep convolutional neural network architecture codenamed Inception V1 for classification and detection on CIFAR-10 dataset.
    * Wide&Deep: jointly trained wide linear models and deep neural networks for recommender systems on Criteo dataset.
* Frontend and User Interface
    * Complete numpy advanced indexing method. Supports value and assignment through tensor index.
    * Some optimizers support separating parameter groups. Different parameter groups can set different `learning_rate` and `weight_decay`.
    * Support setting submodule's logging level independently, e.g. you can set logging level of module `A` to warning and set logging level of module `B` to info.
    * Support weights to be compiled according to shape to solve the problem of large memory overhead.
    * Add some operators implement and grammar support in pynative mode. To be consistent with graph mode.
    * User interfaces change log
      * Learning rate and weight decay making group params([!637](https://gitee.com/mindspore/mindspore/pulls/637))
      * Support weights to be compiled according to shape([!1015](https://gitee.com/mindspore/mindspore/pulls/1015))
      * delete some context param([!1100](https://gitee.com/mindspore/mindspore/pulls/1100))
      * ImageSummary/ScalarSummary/TensorSummary/HistogramSummary([!1329](https://gitee.com/mindspore/mindspore/pulls/1329))([!1425](https://gitee.com/mindspore/mindspore/pulls/1425))
* Executor and Performance Optimization
    * Support doing evaluation while in training process, so that the accuracy of training can be easily obtained.
    * Enable second-order optimization for resnet50, which can achieve 75.9% accuracy in 45 epochs (Resnet50 @ImageNet).
    * Optimize pynative implementation and improve it's execution performance.
    * Optimize summary record implementation and improve its performance.
* Data processing, augmentation, and save format
    * Support simple text processing, such as tokenizer/buildvocab/lookup.
    * Support padding batch.
    * Support split or concat dataset.
    * Support MindDataset reading from file list.

### Other Hardware Support
* GPU platform
    * New models supported: MobileNetV2, MobileNetV3.
    * Support mixed precision training.
    * Support device memory swapping.

## Bugfixes
* Python API
    * An exception to the broadcast input data type check([!712](https://gitee.com/mindspore/mindspore/pulls/712))
    * Fix issues assignsub return value 0([!1036](https://gitee.com/mindspore/mindspore/pulls/1036))
    * Fix issue Conv2dBackpropInput bprop should return 3 instead of 2 items([!1001](https://gitee.com/mindspore/mindspore/pulls/1001))
    * Fix sens shape error of TrainOneStepWithLossScaleCell([!1050](https://gitee.com/mindspore/mindspore/pulls/1050))
    * Fix BatchNormGrad operator([!1344](https://gitee.com/mindspore/mindspore/pulls/1344))
* Executor
    * Fix dropout，topK and addn errors in PyNative mode ([!1285](https://gitee.com/mindspore/mindspore/pulls/1285), [!1138](https://gitee.com/mindspore/mindspore/pulls/1138), [!1033](https://gitee.com/mindspore/mindspore/pulls/1033)).
    * Fix memory leaks after execution in PyNatvie mode ([!1201](https://gitee.com/mindspore/mindspore/pulls/1201)).
    * Fix HCCL failure in some special scenes ([!1204](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/1204), [!1252](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/1252)).
    * Fix SSD network when Select failed, cann't find kernel info([!1449](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/1449)).
    * Fix Topk operator selection strategy bug between aicore and aicpu([!1367](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/1367)).
    * Fix input memory size of 'assign' op unequal in control sink mode when assigning a data from one child graph to another child graph([!802](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/802)).
    * Fix allreduce ir inconsistency([!989](https://gitee.com/mindspore/dashboard/projects/mindspore/mindspore/pulls/989)).
* GPU platform
    * Fix summary for gradient collection ([!1364](https://gitee.com/mindspore/mindspore/pulls/1364))
    * Fix the slice operator ([!1489](https://gitee.com/mindspore/mindspore/pulls/1489))
* Data processing
    * Fix memory problems of GeneratorDataset of sub-process ([!907](https://gitee.com/mindspore/mindspore/pulls/907))
    * Fix getting data timeout when training the cifar10 dataset under the lenet([!1391](https://gitee.com/mindspore/mindspore/pulls/1391))

## Contributors
Thanks goes to these wonderful people:

Alexey Shevlyakov, Amir Lashkari, anthony, baihuawei, biffex, buxue, caifubi, candanzg, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenzomi, chujinjin, cristoval, dengwentao, eric, etone-chan, fary86, gaojing, gengdongjie, gongchen, guohongzilong, guozhijian, heleiwang, hesham, He Wei, Hoai Linh Tran, hongxing, huangdongrun, huanghui, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jonwe, jonyguo, Junhan Hu, Kang, kingfo, kswang, laiyongqiang, leopz, lichenever, lihongkang, limingqi107, liubuyu, liuliyan2, liuwenhao4, liuxiao, liuxiao, liyong, lizhenyu, lvliang, Margaret_wangrui, meixiaowei, ms_yan, Nat Sutyanyong, ougongchang, panfengfeng, panyifeng, Peilin Wang, peixu_ren, qianlong, rick_sanchez, seatea, sheng, shijianning, simson, sunsuodong, Tinazhang, VectorSL, wandongdong, wangcong, wanghua, wangnan39, Wei Luning, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuxuejian, Xiaoda Zhang, xiefangqi, xulei2020, Yang, yangjie159, yangruoqi713, yangyongjie, yangzhenzhang, Yanjun Peng, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yujianfeng, YuJianfeng, yvetteliu, zhangdengcheng, Zhang Qinghua, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, zhouyuanshen, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang

Contributions of any kind are welcome!

# Release 0.2.0-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    * ResNet101: Deep Residual Learning for Image Recognition.

* Frontend and User Interface
   * Support for all python comparison operators.
   * Support for math operators **,//,%. Support for other python operators like and/or/not/is/is not/ in/ not in.
   * Support for the gradients of function with variable arguments.
   * Support for tensor indexing assignment for certain indexing type.
   * Support for dynamic learning rate.
   * User interfaces change log
     * DepthwiseConv2dNative, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropInput([!424](https://gitee.com/mindspore/mindspore/pulls/424))
     * ReLU6, ReLU6Grad([!224](https://gitee.com/mindspore/mindspore/pulls/224))
     * GeneratorDataset([!183](https://gitee.com/mindspore/mindspore/pulls/183))
     * VOCDataset([!477](https://gitee.com/mindspore/mindspore/pulls/477))
     * MindDataset, PKSampler([!514](https://gitee.com/mindspore/mindspore/pulls/514))
     * map([!506](https://gitee.com/mindspore/mindspore/pulls/506))
     * Conv([!226](https://gitee.com/mindspore/mindspore/pulls/226))
     * Adam([!253](https://gitee.com/mindspore/mindspore/pulls/253))
     * _set_fusion_strategy_by_idx, _set_fusion_strategy_by_size([!189](https://gitee.com/mindspore/mindspore/pulls/189))
     * CheckpointConfig([!122](https://gitee.com/mindspore/mindspore/pulls/122))
     * Constant([!54](https://gitee.com/mindspore/mindspore/pulls/54))
* Executor and Performance Optimization
    * Support parallel execution of data prefetching and forward/backward computing.
    * Support parallel execution of gradient aggregation and forward/backward computing in distributed training scenarios.
    * Support operator fusion optimization.
    * Optimize compilation process and improve the performance.
* Data processing, augmentation, and save format
    * Support multi-process of GeneratorDataset/PyFunc for high performance
    * Support variable batchsize
    * Support new Dataset operators, such as filter,skip,take,TextLineDataset

### Other Hardware Support
* GPU platform
    * Use dynamic memory pool by default on GPU.
    * Support parallel execution of computation and communication.
    * Support continuous address allocation by memory pool.
* CPU platform
    * Support for windows 10 OS.

## Bugfixes
* Models
    * Fix mixed precision bug for VGG16 model ([!629](https://gitee.com/mindspore/mindspore/pulls/629)).
* Python API
    * Fix ControlDepend operator bugs on CPU and GPU ([!396](https://gitee.com/mindspore/mindspore/pulls/396)).
    * Fix ArgMinWithValue operator bugs ([!338](https://gitee.com/mindspore/mindspore/pulls/338)).
    * Fix Dense operator bugs on PyNative mode ([!276](https://gitee.com/mindspore/mindspore/pulls/276)).
    * Fix MatMul operator bugs on PyNative mode ([!288](https://gitee.com/mindspore/mindspore/pulls/288)).
* Executor
    * Fix operator selection bugs and make it general ([!300](https://gitee.com/mindspore/mindspore/pulls/300)).
    * Fix memory reuse bug for GetNext op ([!291](https://gitee.com/mindspore/mindspore/pulls/291)).
* GPU platform
    * Fix memory allocation in multi-graph scenarios ([!444](https://gitee.com/mindspore/mindspore/pulls/444)).
    * Fix bias_add_grad under fp16 precision ([!598](https://gitee.com/mindspore/mindspore/pulls/598)).
    * Fix support for fp16 kernels on nvidia 1080Ti([!571](https://gitee.com/mindspore/mindspore/pulls/571)).
    * Fix parsing of tuple type parameters ([!316](https://gitee.com/mindspore/mindspore/pulls/316)).
* Data processing
    * Fix TypeErrors about can't pickle mindspore._c_dataengine.DEPipeline objects([!434](https://gitee.com/mindspore/mindspore/pulls/434)).
    * Add TFRecord file verification([!406](https://gitee.com/mindspore/mindspore/pulls/406)).

## Contributors
Thanks goes to these wonderful people:

Alexey_Shevlyakov, Cathy, Chong, Hoai, Jonathan, Junhan, JunhanHu, Peilin, SanjayChan, StrawNoBerry, VectorSL, Wei, WeibiaoYu, Xiaoda, Yanjun, YuJianfeng, ZPaC, Zhang, ZhangQinghua, ZiruiWu, amongo, anthonyaje, anzhengqi, biffex, caifubi, candanzg, caojian05, casgj, cathwong, ch-l, chang, changzherui, chenfei, chengang, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, dengwentao, dinghao, fanglei, fary86, flywind, gaojing, geekun, gengdongjie, ghzl, gong, gongchen, gukecai, guohongzilong, guozhijian, gziyan, h.farahat, hesham, huangdongrun, huanghui, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, jonathan_yan, jonyguo, jzw, kingfo, kisnwang, laiyongqiang, leonwanghui, lianliguang, lichen, lichenever, limingqi107, liubuyu, liuxiao, liyong, liyong126, lizhenyu, lupengcheng, lvliang, maoweiyong, ms_yan, mxm, ougongchang, panfengfeng, panyifeng, pengyanjun, penn, qianlong, seatea, simson, suteng, thlinh, vlne-v1, wangchengke, wanghua, wangnan39, wangqiuliang, wenchunjiang, wenkai, wukesong, xiefangqi, xulei, yanghaitao, yanghaoran, yangjie159, yangzhenzhang, yankai10, yanzhenxiang2020, yao_yf, yoonlee666, zhangbuxue, zhangz0911gm, zhangzheng, zhaojichen, zhaoting, zhaozhenlong, zhongligeng, zhoufeng, zhousiyi, zjun, zyli2020, yuhuijun, limingqi107, lizhenyu, chenweifeng.

Contributions of any kind are welcome!

# Release 0.1.0-alpha

## Main Features

### Ascend 910 Training and Inference Framework
* Recommended OS: Ubuntu 16.04 (or later) or EulerOS 2.5 or EulerOS 2.8
* Python version: 3.7.5
* Preset models
    * ResNet-50: residual structure-based convolutional neural network (CNN) for image classification, which is widely used.
    * AlexNet: classic CNN for image classification, achieving historical results in ImageNet LSVRC-2012.
    * LeNet: classic CNN for image classification, which was proposed by Yann LeCun.
    * VGG16: classic CNN for image classification, which was proposed by Oxford Visual Geometry Group.
    * YoloV3: real-time object detection network.
    * NEZHA: BERT-based Chinese pre-training network produced by Huawei Noah's Ark Laboratory.
* Execution modes
    * Graph mode: provides graph optimization methods such as memory overcommitment, IR fusion, and buffer fusion to achieve optimal execution performance.
    * PyNative mode: single-step execution mode, facilitating process debugging.
* Debugging capability and methods
    * Save CheckPoints and Summary data during training.
    * Support asynchronous printing.
    * Dump the computing data.
    * Support profiling analysis of the execution process performance.
* Distributed execution
    * Support AllReduce, AllGather, and BroadCast collective communication.
    * AllReduce data parallel: Each device obtains different training data, which accelerates the overall training process.
    * Collective communication-based layerwise parallel: Models are divided and allocated to different devices to solve the problem of insufficient memory for large model processing and improve the training speed.
    * Automatic parallel mode: The better data and model parallel mode can be predicted based on the cost model. It is recommended that this mode be used on ResNet series networks.
* Automatic differentiation
    * Implement automatic differentiation based on Source to Source.
    * Support distributed scenarios and automatic insertion of reverse communication operators.
* Data processing, augmentation, and save format
    * Load common datasets such as ImageNet, MNIST, CIFAR-10, and CIFAR-100.
    * Support common data loading pipeline operations, such as shuffle, repeat, batch, map, and sampler.
    * Provide basic operator libraries to cover common CV scenarios.
    * Support users to customize Python data augmentation operators through the Pyfunc mechanism.
    * Support the access of user-defined datasets through the GeneratorDataset mechanism.
    * Provide the MindSpore data format, data aggregation and storage, random access example, data partition, efficient parallel read, user-defined index, and dataset search.
    * Convert user datasets to the MindSpore data format.
    * After data processing and augmentation, provide training applications in feed and graph modes.
* FP32/16 mixed precision computation, supporting automatic and manual configuration
* Provide common operators such as nn, math, and array, which can be customized.

### Inference Deployment
* Deploy models in MindSpore format on the Ascend 310 platform for inference.
* Save models in ONNX format.
* Support saving models in LITE format and running models based on the lightweight inference framework.
    * Recommended OS: Android 4.3 or later
    * Supported network type: LeNet
    * Provide the generalization operators generated by TVM and operators generated after specific networks are tuned.

### Other Hardware Support
* GPU platform training
    * Recommended OS: Ubuntu 16.04
    * CUDA version: 9.2 or 10.1
    * CuDNN version: 7.6 or later
    * Python version: 3.7.5
    * NCCL version: 2.4.8-1
    * OpenMPI version: 3.1.5
    * Supported models: AlexNet, LeNet, and LSTM
    * Supported datasets: MNIST and CIFAR-10
    * Support data parallel.
* CPU platform training
    * Recommended OS: Ubuntu 16.04
    * Python version: 3.7.5
    * Supported model: LeNet
    * Supported dataset: MNIST
    * Provide only the stand-alone operation version.

## Peripherals and Tools
* [MindSpore Official Website] (https://www.mindspore.cn/)
* [MindInsight Visualization Debugging and Optimization] (https://gitee.com/mindspore/mindinsight)
* [MindArmour Model Security Hardening Package] (https://gitee.com/mindspore/mindarmour)
* [GraphEngine Computational Graph Engine] (https://gitee.com/mindspore/graphengine)
