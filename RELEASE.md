# Release 0.3.0-alpha

## Major Features and Improvements

### TODO

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
