# MindSpore Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore 2.8.0 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] The [mindspore.dataset.dataloader.DataLoader](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.dataloader.html) API is added, which is consistent with the mainstream API usage in the industry in terms of usage and functions, facilitating user reference. In addition, various commonly used methods for `datasets`, `samplers`, `collate functions`, and `tools` are added.
- [STABLE] The `.send(...)` and `.recv(...)` APIs are added for communication between nodes of the dataset object. These APIs enable the dataset data processing results to be transmitted between different nodes.
- [STABLE] When `.map(...)` is used for data augmentation and a custom PyFunc augmentation function is used in multi-process mode, if the PyFunc function execution is slow or hung, a warning will be printed to notify the user: `** worker subprocess stack:...`.

#### Executor

- [STABLE] MindSpore now supports [custom backends](https://www.mindspore.cn/tutorials/en/master/custom_program/custom_backend.html), allowing users to adapt to third-party backends.

#### Compiler

- [STABLE] In graph mode, augmented assignment statements are parsed into corresponding in-place operators, improving graph mode performance and unifying programming experience for pynative and graph mode. Currently only the `Ascend` backend is supported.

#### PyNative

- [STABLE] The [recompute interface](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute) supports setting use_reentrant=False to enable gradient computation for complex type inputs. It also supports setting output_recompute to determine whether outputs should be re-computed.
- [STABLE] Added support for CPU Tensor conversion with [DLPack](https://www.mindspore.cn/docs/en/master/api_python/mindspore.utils.html#mindspore.utils.dlpack.from_dlpack).
- [STABLE] Storage supports shared memory between processes.

### API Change

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) API has added three Native Sparse Attention(NSA) interfaces:

  | mindspore.ops                        |
  | :----------------------------------- |
  | mindspore.ops.nsa_compress           |
  | mindspore.ops.nsa_compress_attention |
  | mindspore.ops.nsa_select_attention   |

- [STABLE] The [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API has added the [mindspore.mint.nn.functional.cosine_embedding_loss](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.cosine_embedding_loss.html#mindspore.mint.nn.functional.cosine_embedding_loss) and [mindspore.mint.nn.CosineEmbeddingLoss](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.CosineEmbeddingLoss.html#mindspore.mint.nn.CosineEmbeddingLoss) interfaces. Most mint interfaces are currently still experimental and outperform ops in terms of performance under Graph Mode O0/O1 and PyNative Mode. O2 compilation mode (Graph Sinking), as well as CPU and GPU backends, are not currently supported and will be improved incrementally.
- [STABLE] The [mindspore.mint.nn.functional.adaptive_max_pool2d](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.adaptive_max_pool2d.html#mindspore.mint.nn.functional.adaptive_max_pool2d) and [mindspore.mint.nn.AdaptiveMaxPool2d](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.AdaptiveMaxPool2d.html#mindspore.mint.nn.AdaptiveMaxPool2d) interfaces have been promoted from demo to stable status.
- [STABLE] The [mindspore.Tensor.view](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.view.html) interface now supports `dtype` as an input argument.
- [STABLE] The [mindspore.mint.nn.functional.interpolate](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.interpolate.html) interface now supports setting `scale_factor` in bilinear/bicubic modes. The restriction preventing `scale_factor` from being set when `align_corners=False` in linear mode has been lifted.
- [STABLE] The [mindspore.ops.grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html) and [mindspore.ops.value_and_grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.value_and_grad.html?highlight=value_and_grad) interfaces add the `sens_param` parameter, which is used to specify whether to configure sensitivity (gradient with respect to the output) in the input.
- [STABLE] When iterating over a dataset, if the output data contains the `string` type, the default output type will change from `Tensor` to `numpy.ndarray`.
- [BETA] Added [mindspore.Tensor.to](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.to.html) and [mindspore.Tensor.to_](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.to_.html), which convert a tensor's device and data type to the specified `device` and `dtype`.
- [BETA] Added [mindspore.Tensor.delete_](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.delete_.html) for actively releasing the memory of the tensor on the `device` or `host`.
- [BETA] Added [mindspore.Tensor.data](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.data.html#mindspore.Tensor.data), providing access to the raw data without tracking its computational history for autograd.

### Backwards Incompatible Change

#### Dataset

- [STABLE] The following obsolete import methods are completely removed: `import mindspore.dataset.vision.c_transforms as c_vision`, `import mindspore.dataset.vision.py_transforms as py_vision`, `import mindspore.dataset.transforms.c_transforms as c_transforms`, `import mindspore.dataset.transforms.py_transforms as py_transforms`.
- [STABLE] The `column_order` parameter in the `.map(...)` operation is completely removed. You can use `.project(...)` to adjust the column order.

### Bug Fixes

#### Dataset

- [ID6JRL](https://gitee.com/mindspore/mindspore/issues/ID6JRL): Fixed an issue where an incorrect data dimension was caused when `GeneratorDataset` (when the custom dataset `__getitem__` returns the self member variable of the `dict` type) and `.batch(batch_size=1)` was used.

### Contributors

anyrenwei,Bellatan,caifubi,Carey,chaijinwei,changzherui,chengbin,chenshan2623,chujinjin,DavidFFFan,DeshiChen,dingjinshan,fangwenyi,fary86,fengyixing,fuchao,Gaoxiong,gaoyong10,guangpengz,guozhijian,haozhang,hedongdong,hhz886,HighCloud,huangbingjian,huangfuxin,huda,jiangna,jiangshanfeng,jiaorui,jijiarong,laoyu,leida,liangchenghui,LiangZhibo,lichen,lijiajie1234,limingqi107,LiNuohang,litingyu,liubuyu,liuchao,liuchuting,liuluobin,liuyanwei,lizhitong,looop5,luochao60,machangwei,maoyuanpeng1,Margaret_wangrui,mengxian,mwt,NaCN,panzhihui,Qiao_Fu,qiuleilei,qqqhhhbbb,r1chardf1d0,rainyhorse,rogeryu11,shaoshengqi,shen_haochen,shenwei41,shuqian0,tanghuikang,Tianci Xiao,tianxiaodong,wang_ziqi,wangjialin,wangyin,wujueying,wusuqin4,wuyanernuo,XianglongZeng,xiaopeng,xiaotianci,xuzhen,yanghaoran,yangting,yao_yf,yide12,yiguangzheng,yuanqi,yuchaojie,Yuheng Wang,YuJianfeng,Yule100,yuliangbin,YzLi,zhangbuxue,Zhanghanbo,zhanghanLeo,zhangyihui,zhangyinxia,zhangzhen,zhaochenjie,zhengzuohe,zhongmin,ZPaC,zyb,zyli2020,阿琛,曹彤,胡彬,宦晓玲,黄勇,李良灿,李林杰,刘飞扬,刘力力,宋佳琪,王振邦,熊攀,徐子康,严珞珈,杨晓春,张峻源,张栩浩

## MindSpore 2.7.2 Release Notes

### Major Features and Improvements

#### Ascend

- [BETA] The GroupedMatMul + Reshape + AssignAdd fuse pattern is supported in static graph mode when jit_level is O1, it can be enabled by setting the environment variable `export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_cluster_ops=GroupedMatmul,Reshape"`.
- [STABLE] Support subhealthy device hot-switch in static graph mode when jit_level is O1 or O1, allowing replacement of subhealthy nodes without restarting the training cluster to achieve minute-level training resumption from breakpoints. Users can enable this feature by setting the environment variable `export MS_ENABLE_TFT="{ARF:1}"`.

### Parallel

- [STABLE] `msrun` supports setting CPU affinity to the scheduler process and worker process via the `taskset` command when launching distributed tasks. Users can set CPU affinity for the launched processes through the input parameter `--bind_core`.

#### PyNative

- [STABLE] Support the `frombuffer` feature on PyNative mode, enabling direct creation of MindSpore tensors from Python buffer protocol objects (such as bytes) without data copying.
- [STABLE] Support the `saved_tensors_hooks` feature in PyNative mode, allowing users to configure custom save and load logic for saved tensors that are retained during the forward pass and consumed during the backward pass.

#### Python

- [STABLE] Supports Python 3.12.

### Compatibility

The CANN 8.5.0 package has completed the upgrade to an open-source architecture and consists of two packages: Toolkit and ops. The decoupling optimization has led to a series of changes in naming conventions, quantities, usage relationships, and installation directories. MindSpore 2.7.2 needs to be used in conjunction with CANN 8.5.0. Compatibility with older versions will continue to be supported in MindSpore 2.8.0.

### Contributors

Bellatan,caifubi,chaijinwei,changzherui,chengbin,fuchao,guangpengz,guozhijian,hhz886,huangbingjian,huda,jiangna,jiaorui,kingxian,liangchenghui,lichen,liubuyu,liuluobin,liuyanwei,looop5,maoyuanpeng1,Margaret_wangrui,mengxian,NaCN,panzhihui,r1chardf1d0,rogeryu11,shaoshengqi,shen_haochen,shenwei41,tanghuikang,wang_ziqi,wangshaocong,wangyin,xiaotianci,xuzhen,yanghaoran,yide12,yuchaojie,yuliangbin,zhangyinxia,zhaochenjie,zhengzuohe,ZPaC,zyli2020,曹彤,范吉斌,胡彬,宦晓玲,李良灿,熊攀

## MindSpore 2.7.1 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] The parallel acceleration capability of the `.map` operation and `.batch` operation in the dataset has been restructured. The original data transfer logic of `c++ thread(with GIL) -> python process pool -> worker[i]` has been optimized to `c++ thread -> msg & shm -> worker[i]`, reducing the data transfer chain and eliminating the dependency on the GIL lock, which significantly improves the efficiency of custom data processing.

#### Parallel

- [STABLE] In static graph mode, the `AlltoAllVC` forward and reverse operators are supported. Users can use this operator through the [mindspore.ops.AlltoAllVC](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.AlltoAllVC.html) interface. In dynamic graph mode, the [mindspore.communication.comm_func.all_to_all_v_c](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.all_to_all_v_c.html) communication interface is supported. Users can use this operator through this interface.
- [STABLE] In dynamic graph mode, supports `TCPStore` function, supports the communication class [mindspore.mint.distributed.TCPStore](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.TCPStore.html), users can use the `TCPStore` function through this class.
- [STABLE] The [mindspore.communication.create_group](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.create_group.html) interface now supports the `hccl_comm` parameter, allowing the reuse of externally created communication groups. Users can create communication groups through other means and pass them to MindSpore for network construction.
- [BETA] The [mindspore.mint.distributed.init_process_group](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.init_process_group.html) interface supports `init_method` or `store` parameters. Users can use this interface to initialize communication in a way that does not depend on the scheduler process.

#### Ascend

- [BETA] Operator automatic fusion is supported in PyNative asynchronous execution mode. Users can enable it by setting the environment variable: `export MS_DEV_PYNATIVE_FUSION_FLAGS="--opt_level=1"`.
- [BETA] The static graph now supports custom operator integration via the CustomOpBuilder method, and exposes the MS_CUSTOM_OPS_REGISTER macro for registering custom operator function classes.
- [BETA] Dynamic graph mode supports the use of custom memory pools via [mindspore.runtime.use_mem_pool](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.use_mem_pool.html) to implement custom memory allocation within the context.

#### PyNative

- [STABLE] The [mindspore.mint.empty_like](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.empty_like.html) and [mindspore.mint.empty](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.empty.html) functions support the `pin_memory` parameter. If users set the `pin_memory` parameter to `True`, the returned Tensor will be allocated on pinned memory. Works only for CPU Tensors.
- [STABLE] View operator performance optimization. By streamlining execution workflows and refining critical implementations, we have significantly enhanced the execution efficiency of View operators in eager mode.
- [STABLE] The [mindspore.Tensor.to](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.to.html) interface is extended to support copying Tensor data between devices.
- [STABLE] Tensor storage supports CPU, GPU, and Ascend. Before 2.7.1, Tensor storage only supported Ascend hardware.

#### Tools

- [STABLE] The static graph supports the joint detection scheme for feature values. Users can enable this feature by configuring the environment variable MS_NPU_ASD_CONFIG.
- [STABLE] HCCS link fault detection is supported. The [stress_detect](https://www.mindspore.cn/docs/en/master/api_python/mindspore.utils.html#mindspore.utils.stress_detect) interface now includes an HCCS detection type.

### API Change

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API provides some new functional and Tensor interfaces. Most of the mint interfaces are currently still experimental interfaces. They perform better than ops interfaces in `jit_level="O0"/"O1"` and pynative mode. Currently, the graph sinking mode and CPU/GPU backend are not supported, and it will be gradually improved in the future.

  | mindspore.mint      |
  | :------------------ |
  | mindspore.mint.real |
  | mindspore.mint.imag |

  | mindspore.Tensor                 |
  | -------------------------------- |
  | mindspore.Tensor.sign_           |
  | mindspore.Tensor.masked_scatter_ |
  | mindspore.Tensor.index_copy_     |
  | mindspore.Tensor.index_fill_     |
  | mindspore.Tensor.sigmoid_        |

- [STABLE] [mindspore.mint.nn.functional.conv1d](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.conv1d.html) and [mindspore.mint.nn.Conv1d](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.Conv1d.html) interfaces have transitioned from demo to stable.
- [STABLE] [mindspore.mint.stack](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.stack.html) and [mindspore.mint.concat](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.concat.html) interfaces now support Tensor inputs of different data types.
- [BETA] The [mindspore.ops.Morph](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Morph.html) operator primitive now supports custom backward propagation functions (bprop), allowing users to define corresponding gradient computation logic for the custom forward function.
- [BETA] [mindspore.recompute](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.recompute.html) supports being used in static graph, user can call the api in the function which is decorated by @jit.
- [BETA] [mindspore.enable_dynamic](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.enable_dynamic.html) supports symbolic deduction, meaning that the input shapes are allowed to use [mindspore.Symbol](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Symbol.html).

### Contributors

Bellatan,caifubi,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,Dring,ehaleva,fary86,gaoyong10,guangpengz,GuoZhibin,guozhijian,haozhang,hedongdong,Henry Shi,hhz886,huangbingjian,huangzhuo,huangziling,huda,Huilan Li,huoxinyou,jiangshanfeng,jiaorui,jiaxueyu,jizewei,kairui_kou,kingxian,kisnwang,leida,liangchenghui,lichen,limingqi107,Linhai,LiNuohang,linux,liubuyu,liudongxu,liuluobin,liuyanwei,lizhitong,looop5,luochao60,machenggui,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,Metaqiang,NaCN,nepdada,panzhihui,Qiao_Fu,qiuleilei,qqqhhhbbb,qujianwei,shaoshengqi,shen_haochen,shenwei41,shuqian0,tanghuikang,uuhuu,wang_ziqi,wangjialin,wujueying,XianglongZeng,Xiaoda,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yanghaoran,yao_yf,yefeng,yide12,yiguangzheng,YijieChen,yuanqi,yuchaojie,YuJianfeng,YukioZzz,yuliangbin,yyyyrf,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhunaipan,ZPaC,zyli2020,范吉斌,龚昊宇,胡彬,宦晓玲,李栋,李良灿,李林杰,刘崇鸣,刘飞扬,刘力力,刘子涵,宋佳琪,孙昊辰,王泓皓,王振邦,俞涵,张栩浩,张学同,周一航

## MindSpore 2.7.0 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] The [mindspore.dataset.vision.read_video](https://www.mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.read_video.html) interface now supports video decoding based on Ascend VDEC. Users can specify the video processing backend as `CPU` (Default) or `Ascend` through the [mindspore.dataset.config.set_video_backend](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.config.set_video_backend.html) interface.
- [STABLE] Add [mindspore.dataset.vision.VideoDecoder](https://www.mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.VideoDecoder.html) interface which provides the ability to decode frames at the given indices and retrieve metadata. Currently only the `Ascend` backend is supported.

#### Parallel

- [STABLE] MindSpore now supports gather and reduce operations for tensors of non-uniform sizes. Users can use this functionality through the [mint.distributed.all_gather_into_tensor_uneven](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.all_gather_into_tensor_uneven.html) and [mint.distributed.reduce_scatter_tensor_uneven](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.reduce_scatter_tensor_uneven.html) interfaces. [mint.distributed.all_gather](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.all_gather.html) and [mint.distributed.reduce_scatter](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.reduce_scatter.html) also support gather and reduce operations for non-uniform sizes, respectively.
- [BETA] The pipeline parallelism supports ZeroBubbleV scheduling, reducing pipeline parallel bubbles, and can be combined with overlap of forward and backward computation-communication phases, to enhance the proportion of communication and computation overlap.
- [Stable] MindSpore optimizes the PP communication domain and significantly reduces HCCL_BUFFER_SIZE.
- [BETA] MindSpore supports fine-grained optimization of the `HCCL_BUFFER_SIZE`, which can be set through the environment variable  `MS_DEV_HCCL_CONF`. Refer to [Environment Variables](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html) for details.

#### Compiler

- [BETA] Support [mindspore.nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) registration of forward hooks and backward hooks in graph mode.

#### Runtime

- [STABLE] MindSpore supports reserving huge page memory. Users can enable this feature by passing the `huge_page_reserve_size` parameter in the [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_memory.html) API.

### Lite

MindSpore Lite​​ delivers lightweight AI inference acceleration capabilities for diverse hardware devices, empowering smart applications. It provides developers with an ​​end-to-end solution​​ and offers algorithm engineers and data scientists a ​​user-friendly development experience​​ characterized by efficient execution and flexible deployment.

To better foster the thriving development of the AI software and hardware application ecosystem, ​​MindSpore Lite has established an independent code repository to drive ecosystem growth​​. In the future, MindSpore Lite will work together with the ​​MindSpore AI community​​ enrich the AI software and hardware application ecosystem.

For further details, please visit the [MindSpore Lite Code Repository](https://gitee.com/mindspore/mindspore-lite).

### API Change

- [STABLE] As part of the task of [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API integration task, the interface definitions and functionalities of several Tensor APIs have been aligned and optimized.

  | mindspore.Tensor                |
  | ------------------------------- |
  | mindspore.Tensor.masked_scatter |
  | mindspore.Tensor.bernoulli_     |
  | mindspore.Tensor.zero_          |
  | mindspore.Tensor.copy_          |

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) API provides a new interface [mindspore.ops.ring_attention_update](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ring_attention_update.html). Currently, it is only supported on Atlas A2 Training Series Products.
- [STABLE] Provide new interface [mindspore.enable_dynamic](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.enable_dynamic.html) to specify whether the shape of the parameter is dynamic shape or dynamic rank.

### Contributors

Bellatan,caifubi,ccsszz,chaijinwei,chengbin,chenweifeng,chujinjin,DavidFFFan,DeshiChen,dingjinshan,fary86,fuchao,gaoyong10,GuoZhibin,guozhijian,haozhang,hedongdong,Henry Shi,hhz886,huangbingjian,huangziling,huda,Huilan Li,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,leida,lichen,limingqi107,LiNuohang,linux,liubuyu,liuluobin,looop5,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,NaCN,One_East,panzhihui,Qiao_Fu,qiuleilei,qiuyufeng,r1chardf1d0,SaiYao,shaoshengqi,shen_haochen,shenwei41,shuqian0,St.Universe,suteng,TAJh,tanghuikang,tianxiaodong,wang_ziqi,wangyibo,wujueying,wusimin,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yanghaoran,yangyingchun,yide12,yonibaehr,yuanqi,yuchaojie,YuJianfeng,YukioZzz,yuliangbin,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhaochenjie,Zhi Feng Wu,zhuguodong,ZPaC,zyli2020,程超,胡犇,胡彬,宦晓玲,黄勇,李良灿,李林杰,刘飞扬,刘勇琪,刘子涵,王振邦,熊攀,杨卉,俞涵,云骑士,张栩浩,周一航

## MindSpore 2.7.0-rc1 Release Notes

### Major Features and Improvements

#### Ascend

- [STABLE] Added GroupedMatmul optimization in O1 scenario, which supports fusion of element-wise operators to significantly reduce data handling overhead and improve computational efficiency. Users can turn it on by setting the environment variable `MS_DEV_GRAPH_KERNEL_FLAGS` to `"--enable_cluster_ops=GroupedMatmul"`.
- [STABLE] Improved ease of use of memory tracker: Users can import tracker data for model memory analysis via [mindspore.runtime.memory_replay(file_path)](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.memory_replay.html); set the tracker data storage path by setting `export MS_ALLOC_CONF=memory_tracker_path:file_path`；reduce the size of saved data by setting `export MS_ALLOC_CONF=simple_tracker:true` to save only the last user of each memory block.
- [STABLE] Optimized the custom operator function of ops.Custom primitive for aclnn types in graph mode, with full support for inputs of non-Tensor type and automatic loading of RegInfo information, which significantly improves the ease of use and flexibility of aclnn custom operator.

#### PyNative

- [STABLE] [mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook) and [mindspore.nn.Cell.register_forward_pre_hook](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook) added the `with_kwargs` argument (default: False) to support passing keyword arguments from the `construct` call to `hook_fn`.
- [STABLE] [mindspore.Tensor.register_hook](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook) now supports registering hooks on output tensors of operators with multiple outputs.
- [STABLE] Enhanced Cell custom bprop. Support verification and automatic conversion of data type and shape for return value and corresponding input.
- [STABLE] Added [Storage](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.storage.html) API for Tensor: Supports memory operations through Storage and related interfaces to achieve memory optimization.
- [STABLE] Added C++ [ms::pynative::PyboostRunner](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/cpp_api_for_custom_ops.html#class-pyboostrunner) interface to facilitate customization of the operator to support PyNative's multi-stage pipelining runtime.

#### parallel

- [STABLE] Configurable HCCL buffer size for communucation group: [communication.create_group](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) supports additional configuration `options` in passing communication domain, sets the communication domain HCCL communication domain cache size by `hccl_config` key-value pair to avoid OOM under multiple communication domains.
- [STABLE] Supported CPU communication interfaces [mint.distributed.all_reduce](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.all_reduce.html)/[mint.distributed.barrier](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.barrier.html)/[mint.distributed.send](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.send.html)/[mint.distributed.recv](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.recv.html), through which users can use the corresponding collective communication operators functions.
- [STABLE] In static graph mode, forward and reverse operators, [AllGatherV](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.AllGatherV.html) and [ReduceScatterV](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceScatterV.html) are supported.
- [BETA] Support for data caching via member methods of [mint.distributed.TCPStore](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.TCPStore.html) function.

#### Training

- [STABLE] Recomputation communication overlap: Supports mutual overlap of the communication between two cells for full recomputation, improving the performance of the recomputation scenario model.
- [STABLE] Quick recovery from accuracy failure without reboot: Supports resuming training of loading a checkpoint file without restrarting the process when training result excepition occurs.
- [STABLE] Silent data corruption detection: Supporting validation of `MatMul` results during forward and backward computations. User can enable it by setting the environment variable `MS_SDC_DETECT_ENABLE` to `1`. Using interfaces in the `mindspore.utils.sdc_detect` module to start/stop detection and get the detection result.

#### Inference

- [STABLE] The vLLM MindSpore plugin has now been adapted to vLLM v0.8.3 and supports foundational features of the vLLM V1 new architecture, including inference capabilities such as Chunked Prefill and Automatic Prefix Caching. For service-oriented deployment, vLLM MindSpore adds support for hybrid DP/TP/EP parallel inference capabilities on DeepSeek-V3/R1, effectively improving both full and incremental inference efficiency while reducing device memory overhead.

#### Tool

MindInsight will no longer update or release new versions after version 2.3，and the related documents have been removed. The origin system optimization data visualization has been integrated into MindStudio Insight, and scalar visualization, parameter distribution visualization, and computational graphs visualization have been integrated into the MindStudio Insight plugins. For details, see the [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html).

- [STABLE] MindSpore Profiler supports msMonitor, enabling users to collect performance data through online monitoring tools.
- [STABLE] MindSpore Profiler adds the record_shapes parameter, supporting users to collect shapes of operators issued by the framework side.
- [STABLE] MindSpore Profiler adds sys resource parameters, supporting the ability to collect sys resource class information.
- [STABLE] MindSpore Profiler adds host_sysparameter, supporting the ability to collect host information such as system call class, storage class, CPU information, etc.
- [STABLE] MindSpore Profiler mstx module provides domain functionality, supporting users to finely control mstx data.

### API Change

- [STABLE] Some of the functional, nn and Tensor interfaces in the DEMO state in the [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API are turned to STABLE. The mint interfaces are still mostly experimental, with better performance than ops interfaces in the graph compilation mode O0/O1 and PyNative mode. Currently O2 compilation mode (graph sink) and CPU, GPU backend are not supported, which will be gradually improved.

  | mindspore.mint              |
  | :-------------------------- |
  | mindspore.mint.randomperm   |
  | mindspore.mint.randn        |
  | mindspore.mint.randint      |
  | mindspore.mint.triu         |
  | mindspore.mint.empty_like   |
  | mindspore.mint.empty        |
  | mindspore.mint.floor_divide |

  | mindspore.mint.nn              |
  | :----------------------------- |
  | mindspore.mint.nn.BatchNorm1d  |
  | mindspore.mint.nn.BatchNorm2d  |
  | mindspore.mint.nn.BatchNorm3d  |
  | mindspore.mint.nn.PixelShuffle |
  | mindspore.mint.nn.Threshold    |

  | mindspore.mint.nn.functional               |
  | :----------------------------------------- |
  | mindspore.mint.nn.functional.threshold     |
  | mindspore.mint.nn.functional.threshold_    |
  | mindspore.mint.nn.functional.pixel_shuffle |

  | mindspore.Tensor              |
  | ----------------------------- |
  | mindspore.Tensor.new_full     |
  | mindspore.Tensor.new_empty    |
  | mindspore.Tensor.floor_divide |
  | mindspore.Tensor.exponential_ |

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) API provides an new interface [mindspore.ops.swiglu](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.swiglu.html). Currently, only Ascend backend is supported.
- [STABLE] [mindspore.ops.svd](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd) of [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) API now extra supports Ascend backend.
- [STABLE] [mindspore.mint.nn.functional.silu](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.silu.html#mindspore.mint.nn.functional.silu) and [mindspore.mint.nn.SiLU](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.SiLU.html) now support input argument `inplace`.
- [STABLE] [communication.create_group](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) adds support for additional configuration `options` for communication domains. HCCL backend supports setting `hccl_config` in `options` to set the HCCL communication domain cache size for communication domains.
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/en/master/api_python/mindspore.runtime.html) API adds implementation of [mindspore.runtime.empty_cache](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.empty_cache.html#mindspore.runtime.empty_cache).
- [STABLE] [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_memory.html#mindspore.runtime.set_memory) now supports input argument `huge_page_reserve_size`.
- [STABLE] [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html) now supports input argument `module_to_cpu_dict`.
- [STABLE] [minspore.nn.cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) added the function to view/save model's state_dict. New interfaces are as follows:

  | mindspore.nn.Cell                       |
  | --------------------------------------- |
  | cell.register_state_dict_post_hook      |
  | cell.register_state_dict_pre_hook       |
  | cell.state_dict                         |
  | cell.register_load_state_dict_pre_hook  |
  | cell.register_load_state_dict_post_hook |
  | cell.load_state_dict                    |

- [STABLE] [minspore.nn.cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) added the function to view/register model's buffer. New interfaces are as follows:

  | mindspore.nn.Cell    |
  | -------------------- |
  | cell.register_buffer |
  | cell.get_buffer      |
  | cell.get_sub_cell    |
  | cell.named_buffer    |
  | cell.buffers         |

#### Backwards Incompatible Change

- [runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)

  The type of `affinity_cpu_list` changed from dictionary to list to customize the configuration of affinity CPU range segments for a single process only. Added a new parameter `module_to_cpu_dict` to support customized configuration of CPU affinity policies for hot module threads.

  | 2.6     | 2.7       |
  | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, {"device0": ["10-19", "23-40"]}) | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, ["10-19", "23-40"],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {"main": [0,1,2,3],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "runtime": [4,5,6],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "pynative": [7,8,9]}) |

### Contributors

baochong,Bellatan,BJ-WANG,caifubi,caiji_zhang,Carey,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,dingjinshan,Dring,ehaleva,Erpim,fary86,fengtingyan,fengyixing,fuchao,gaoyong10,gengdongjie,guangpengz,GuoZhibin,gupengcheng0401,haozhang,hedongdong,hhz886,huandong1,huangbingjian,huangziling,huda,HuilanLi,hujiahui8,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,jshawjc,kairui_kou,kingxian,kisnwang,lanzhineng,leida,LiangZhibo,lichen,limingqi107,LiNuohang,linux,liubuyu,liuchengji,liuluobin,liuyanwei,lkp,looop5,lujiale,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,NaCN,One_East,panshaowu,panzhihui,pengqi,Qiao_Fu,qiuleilei,qiuyufeng,rainyhorse,SaiYao,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shuqian0,tanghuikang,tangmengcheng,tan-wei-cheng,tianxiaodong,uuhuu,wang_ziqi,WangChengzhao,wangshaocong,wangyibo,wujueying,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yangguodong,yanghaoran,yangyingchun,Yanzhi_YI,yide12,yihangchen,YijieChen,yuanqi,yuchaojie,yuezenglin,YuJianfeng,YukioZzz,yuliangbin,yyuse,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhouyaqiang0,zhunaipan,zichun_ye,ZPaC,zyli2020,程超,范吉斌,胡犇,胡彬,宦晓玲,黄勇,李栋,李良灿,李林杰,李寅杰3,刘飞扬,刘力力,刘勇琪,刘子涵,梅飞要,宋佳琪,王泓皓,王禹程,王振邦,熊攀,徐安越,杨卉,杨明海,俞涵,虞良斌,云骑士,张栩浩,周一航

## MindSpore 2.6.0 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] The sharding sampling behavior of the [MindDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.MindDataset.html) interface has been changed from block-based sampling (Data sharding strategy 2 in the link) to interval sampling (Data sharding strategy 1 in the link). Users can control whether to switch back to block-based sampling by setting the MS_DEV_MINDRECORD_SHARD_BLOCK environment variable.
- [STABLE] GeneratorDataset supports spawn to start multiprocessing, and supports the use of Ascend back-end data augmentation methods in multiprocessing. Users can set [mindspore.dataset.config.set_multiprocessing_start_method("spawn")](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.config.set_multiprocessing_start_method.html) to enable multiprocessing in spawn mode.
- [STABLE] The `shuffle` parameter in [MindDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.MindDataset.html) supports the `Shuffle.ADAPTIVE`option, which adaptively adjusts the shuffle sample count strategy based on the number of samples to reduce training memory overhead and lower the risk of OOM. If global shuffle is desired, users can specify `Shuffle.GLOBAL`, but they must ensure sufficient machine memory.

#### Ascend

- [STABLE] In MindSpore's dynamic graph mode, the AscendC custom operators integrated by the [ops.Custom](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Custom.html) primitive support multiple output types, and `ops.Custom` supports type inference on the C++ side.
- [BETA] In MindSpore's dynamic graph mode, added `CustomOpBuilder` to support online building and loading of custom operators.
- [STABLE] When using the O1 compilation option, users can control the scope of graph and computation fusion optimization. Users can enable or disable specific fusion patterns by setting the environment variable MS_DEV_GRAPH_KERNEL_FLAGS with options such as enable_fusion_pattern_only or disable_fusion_pattern. Additionally, it supports reading configuration from a file via the --path=example.json option.
- [STABLE] Support users to set the aclop operator cache information aging configuration and error message reporting mode configuration through the [mindspore.device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/en/master/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) interface.
- [STABLE] GE backend only supports whole graph sinking and lazy inline subgraph sinking, while other scenarios are no longer supported.
- [BETA] In MindSpore's static graph O0/O1 mode, `mindpore.nn.Cell` adds the new interface `offload` and the attribute `backward_prefetch`. Users can use this interface through [Cell.offload(backward_prefetch)](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.offload) to offload activations within a specific `Cell` class from the device side to the host side during the forward training phase, and prefetch activations from the host side to the device side during the backward training phase.

#### Parallel

- [STABLE] Parallel pdb debugging, dynamic and static graph mode are supported. dynamic graph mode is recommended.
- [STABLE] New API [mindspore.communication.get_comm_name](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.get_comm_name.html), which allows users to query the name of the underlying communicator of the HCCL collection communication library.
- [STABLE] Added [AutoParallel](https://www.mindspore.cn/docs/en/master/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html) API to support parallel configuration of individual networks, solving the problem of excessive scope of parallel configuration.
- [STABLE] SeqPipe now supports two new scheduling methods, seqvpp and seqsmartvpp, significantly reducing the memory cost in scenarios where SeqPipe is combined with VPP.
- [STABLE] Static graph now supports zero2/zero3 level memory optimization, reducing the memory cost for models that require pure data parallel (DP) training.
- [STABLE] Static graph now supports 1b1f compute and communication overlapping in pipeline parallelism conditions, enhancing the performance of pipeline parallelism.
- [STABLE] Static graphs support grad model parallel communication overlap with dw computation under tensor model parallelism and expert model parallelism, improving model training performance.
- [STABLE] Static graph auto-parallel strategy propagation mode is updated to prioritize the layout propagation to improve the accuracy.
- [STABLE] Static graph auto-parallel support using [mindspore.parallel.shard](https://www.mindspore.cn/docs/en/master/api_python/parallel/mindspore.parallel.shard.html) interface to configure strategies for mint operators, optimized for multi-input operators.
- [STABLE] For LLM reinforcement learning， now we support DP/MP/PP for training and inferenceing phase.
- [STABLE] MindSpore supports users to query whether the distributed module is available and whether the communication module is initialized. Users can query whether the distributed module is available through the [mint.distributed.is_available](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.is_available.html) interface, and query whether the communication module is initialized through the [mint.distributed.is_initialized](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.is_initialized.html) interface.
- [STABLE] MindSpore static graph mode supports the `AlltoAllV` forward and reverse operators. Users can use this operator through the [ops.AlltoAllV](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.AlltoAllV.html) interface.
- [STABLE] Support CPU operators [mindspore.mint.distributed.allreduce](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.all_reduce.html#mindspore.mint.distributed.all_reduce), [mindspore.mint.distributed.barrier](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.barrier.html#mindspore.mint.distributed.barrier), [mindspore.mint.distributed.send](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.send.html#mindspore.mint.distributed.send), and [mindspore.mint.distributed.recv](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.distributed.recv.html#mindspore.mint.distributed.recv), and the users can use the corresponding aggregate communication operator functions through these interfaces.

#### Inference

- [STABLE] Support full-precision inference with BFloat16 and quantized inference with W8A8 for DeepSeek-V3/R1. Add or optimize 12 fusion operators including RmsNormQuant, MatMul+Sigmoid+Add, and Transpose+BatchMatMul+Transpose to enhance the inference performance of DeepSeek-V3/R1.
- [BETA] Support deploying inference services of DeepSeek-V3/R1 using MindIE and MindSpore Transformers large model development suite.
- [STABLE] Optimize the process of loading safetensors and realize on-demand initialization of GE, which reduces both memory usage and startup time when deploying inference services using MindIE and MindSpore Transformers large model suite.
- [BETA] Support deploying inference services of DeepSeek-V3/R1 and Qwen2.5 using the [vLLM-MindSpore](https://gitee.com/mindspore/vllm-mindspore) plugin and vLLM v0.6.6.post1.

#### Profiler

- [STABLE] The MindSpore framework supports obtaining communication domain parallel strategy information, which can be visualized to improve performance troubleshooting efficiency in cluster scenarios.
- [STABLE] MindSpore Profiler dynamic profiling supports lightweight instrumentation, allowing users to dynamically enable lightweight tracing and view performance data in real time.
- [STABLE] MindSpore Profiler's lightweight instrumentation capability has been enhanced, supporting key phases such as dataloader and save checkpoint with lightweight tracing information.
- [STABLE] Profiler supports viewing memory_access related aicore metric information.
- [STABLE] MindSpore Profiler supports [mindspore.profiler.profile](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler.profile.html) and [_ExperimentalConfig](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler._ExperimentalConfig.html), as well as the [tensorboard_trace_handler](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html) parameter, improving tool usability.
- [STABLE] MindSpore Profiler dynamic profiling now supports memory data collection, allowing users to dynamically enable memory data gathering to enhance tool usability.

#### Compiler

- [BETA] The graph mode supports the inplace and view operator forward expression capabilities.
- [BETA] Add new operator primitive [mindspore.ops.Morph](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Morph.html) in GRAPH mode, enabling encapsulation of user-defined function as operator primitive [mindspore.ops.Morph](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Morph.html), facilitating encapsulation of irregular collective communication operations (such as [mindspore.ops.AlltoAllV](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.AlltoAllV.html)) for distributed auto-parallel training scenarios.

### API Change

#### New APIs & Enhanced APIs

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API provides more functional, nn interfaces. The mint interface is currently an experimental interface and performs better than ops in `jit_level="O0"` and pynative mode. Currently, the graph sinking mode and CPU/GPU backend are not supported, and it will be gradually improved in the future.

  | mindspore.mint                  |
  | :------------------------------ |
  | mindspore.mint.reshape          |
  | mindspore.mint.triangular_solve |
  | mindspore.mint.index_add        |
  | mindspore.mint.logaddexp2       |
  | mindspore.mint.diag             |

  | mindspore.mint.nn              |
  | :----------------------------- |
  | mindspore.mint.nn.Sigmoid      |
  | mindspore.mint.nn.Conv2d       |
  | mindspore.mint.nn.PixelShuffle |

  | mindspore.mint.nn.functional                     |
  | :----------------------------------------------- |
  | mindspore.mint.nn.functional.adaptive_avg_pool3d |
  | mindspore.mint.nn.functional.conv2d              |
  | mindspore.mint.nn.functional.avg_pool3d          |
  | mindspore.mint.nn.functional.elu_                |
  | mindspore.mint.nn.functional.pixel_shuffle       |

  | others                   |
  | ------------------------ |
  | mindspore.mint.optim.SGD |
  | mindspore.mint.linalg.qr |

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) API also provides some new stable interfaces. Besides, some demo interfaces are changed into stable ones.

  | mindspore.mint           |
  | :----------------------- |
  | mindspore.mint.full_like |
  | mindspore.mint.log2      |
  | mindspore.mint.isneginf  |

  | mindspore.mint.nn           |
  | :-------------------------- |
  | mindspore.mint.nn.GLU       |
  | mindspore.mint.nn.KLDivLoss |

  | mindspore.mint.nn.functional        |
  | :---------------------------------- |
  | mindspore.mint.nn.functional.glu    |
  | mindspore.mint.nn.functional.kl_div |

  | mindspore.Tensor          |
  | :------------------------ |
  | mindspore.Tensor.isneginf |
  | mindspore.Tensor.log2     |

- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API provides more Tensor methods. Currently, these Tensor methods are experimental interfaces and currently does not support the graph sink mode and CPU, GPU backend, and they will be gradually improved in the future. Details can be found in [API list](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) in official website.
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) provides two inference API [mindspore.ops.moe_token_permute](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.moe_token_permute.html#mindspore.ops.moe_token_permute) and [mindspore.ops.moe_token_unpermute](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.moe_token_unpermute.html#mindspore.ops.moe_token_unpermute). Currently, only Ascend backend is supported.
- [STABLE] [mindspore.mint.nn.functional.gelu](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.functional.gelu.html) and [mindspore.mint.nn.GeLU](https://www.mindspore.cn/docs/en/master/api_python/mint/mindspore.mint.nn.GELU.html) now support input argument "approximate".
- [STABLE] Added the offline parsing interface [mindspore.profiler.profiler.analyse](https://gitee.com/link?target=https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler.profiler.analyse.html).

#### Backwards Incompatible Change

- For [mindspore.ops.Xlogy](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Xlogy.html), the arguments `input` and `other` no longer support non-tensor input. [(!81625)
  ](https://gitee.com/mindspore/mindspore/pulls/81625)

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  ops.Xlogy(input [Tensor, numbers.Number, bool],
            other [Tensor, numbers.Number, bool])
  </td>
  <td><pre>
  ops.Xlogy(input [Tensor], other [Tensor])
  </td>
  </tr>
  </table>

- `&` operator no longer supports the input Tensor with data type of uint32/uint64 on Ascend backend in PyNative mode.  `^` operator no longer supports the  input Tensor with data type of uint16/uint32/uint64 on Ascend backend in PyNative mode. `|` operator no longer supports the input Tensor with data type of uint16/uint32/uint64 on Ascend backend in PyNative mode at the scene of `tensor | scalar`. [(!82054)](https://gitee.com/mindspore/mindspore/pulls/82054)
- `%` operator no longer supports the input Tensor with data type of uint16/uint32/uint64 on CPU and GPU backend. [(!83055)](https://gitee.com/mindspore/mindspore/pulls/83055)
- [mindspore.jit](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html) interface parameter change。[(!80248)](https://gitee.com/mindspore/mindspore/pulls/80248)

  The name of parameter `fn` is changed to `function` .

  Remove parameter `mode` , `input_signature` , `hash_args` , `jit_config` and `compile_once` .

  Add parameter `capture_mode` to set how to compile to MindSpore graph.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(mode="PIJit")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(capture_mode="bytecode")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `jit_level` to set the level of compilation optimization.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(jit_level="O0"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_level="O0")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `dynamic` to set whether dynamic shape compilation should be performed.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(dynamic=1)
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `fullgraph` to set whether to capture the entire function into graph.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(jit_syntax_level="STRICT"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(fullgraph=True)
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `backend` to set the compilation backend to be used.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(backend="ms_backend")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `options` to set the dictionary of options to pass to the compilation backend.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(infer_boost="on"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(infer_boost="on")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

- The `mindspore.profiler.tensor_board_trace_handler` interface change.

  The `mindspore.profiler.tensor_board_trace_handler` interface is now renamed to [mindspore.profiler.tensorboard_trace_handler](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html).

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> from mindspore.profiler import tensor_board_trace_handler
  </pre>
  </td>
  <td><pre>
  >>> from mindspore.profiler import tensorboard_trace_handler
  </pre>
  </td>
  </tr>
  </table>

- The  `mindspore.set_context` interface change。

  The `exception_dump` field in the `ascend_config` parameter was changed to the `"dump"` field in [device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/en/master/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html).

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.set_context(ascend_config = {"exception_dump": "2"})
  </pre>
  </td>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.device_context.ascend.op_debug.aclinit_config({"dump": {"dump_scene": "lite_exception"}})
  </pre>
  </td>
  </tr>
  </table>

- The printing content of `mindspore.Tensor` change。

  The original Tensor prints only the value, while the new Tensor prints key information such as shape and dtype.

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> import mindspore as ms
  >>> tensor = ms.Tensor([1,1,1], dtype=ms.float32)
  >>> print(tensor)
  [1. 1. 1.]
  </pre>
  </td>
  <td><pre>
  >>> import mindspore as ms
  >>> tensor = ms.Tensor([1,1,1], dtype=ms.float32)
  >>> print(tensor)
  Tensor(shape=[3], dtype=Float32, value= [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00])
  </pre>
  </td>
  </tr>
  </table>

- In graph mode, Ascend backend, when jit_level is O2, the Dump interface changes.

  In the graph Ascend backend jit_level O2 scenario, the environment variables `MINDSPORE_DUMP_CONFIG` and `ENABLE_MS_GE_DUMP` have been deprecated, and the dump-related functions have been migrated to the msprobe tool. For more details, please refer to [msprobe Tool MindSpore Scene Accuracy Data Collection Guide](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md).

### Contributors

amyMaYun,Ava,baishanyang,br_fix_save_strategy_ckpt,caifubi,caoxubo,cccc1111,ccsszz,chaijinwei,chaiyouheng,changzherui,chengbin,chopupu,chujinjin,congcong,dairenjie,DavidFFFan,DeshiChen,dingjinshan,fary86,fengqiang,fengyixing,ffmh,fuhouyu,Gallium,gaoshuanglong,gaoyong10,geyuhong,guoyq16,guoyuzhe,GuoZhibin,guozhijian,gupengcheng0401,hangq,Hanshize,haozhang,hedongdong,hhz886,HighCloud,horcham,huangbingjian,huangxiang360729,huangzhichao2023,huangzhuo,huangziling,huda,Huilan Li,hujiahui8,huoxinyou,jiangchao_j,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,JoeyLin,jshawjc,kairui_kou,kakyo82,kisnwang,leida,lianghongrui,LiangZhibo,LiaoTao_Wave,lichen,limingqi107,LiNuohang,linux,litingyu,liubuyu,liuchuting,liuluobin,liuyanwei,LLLRT,looop5,luochao60,luojianing,luoxuewei,luoyang,lyk,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mylinchi,NaCN,panzhihui,pengqi,PingqiLi,pipecat,qiuyufeng,qiuzhongya,qiwenlun,r1chardf1d0,rachel0858,rainyhorse,Rudy_tan,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shiro-zzz,shuqian0,stavewu,TAJh,tanghuikang,tangmengcheng,tongdabao,TuDouNi,VectorSL,wang_ziqi,wangjie,wangliaohui97,wangpingan,wangyibo,weiyang,wja,wudongkun,wujueying,wuweikang,wwwbby,xfan233,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,xiedejin1,XinDu,xuxinglei,xuzhen,xuzixiang,yang guodong,yangben,yanghaoran,yangruoqi713,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yide12,yihangchen,YijieChen,yonibaehr,Youmi,yuanqi,yuchaojie,yuezenglin,Yuheng Wang,YuJianfeng,YukioZzz,ZeyuHan,zhaiyukun,Zhang QR,zhangbuxue,zhangdanyang,zhangshucheng,zhangyinxia,ZhangZGC,zhangzhen,zhengzuohe,zhouyaqiang0,zichun_ye,zlq2020,zong_shuai,ZPaC,zyli2020,舛扬,范吉斌,冯一航,胡犇,胡彬,宦晓玲,简云超,李栋,李良灿,李林杰,李寅杰3,刘思铭,刘勇琪,刘子涵,梅飞要,任新,十一雷,孙昊辰,王泓皓,王禹程,王振邦,熊攀,俞涵,虞良斌,云骑士,张栩浩,赵文璇,周一航

## MindSpore Lite 2.6.0 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite supports configuring operator parallel inference acceleration during model conversion. You only need to configure the stream_label_file option during model conversion to specify the operators that need parallel inference.
- [STABLE] MindSpore Lite supports the conversion of onnx if operators in the Ascend backend.

### API Change

- [STABLE] In the acl model conversion configuration, a new stream_label_file option is added under the ascend_context option to enable multi-stream parallel inference.

### Contributors

熊攀,ZhangZGC,yanghaoran,李林杰,shenwei41,xiaotianci,panzhihui,guozhijian,胡彬,tangmengcheng,XianglongZeng,cccc1111,stavewu,刘思铭,r1chardf1d0,jiangshanfeng

## MindSpore 2.5.0 Release Notes

### Major Features and Improvements

#### Distributed Startup Component msrun

- [STABLE] msrun supports passing in the hostname of the node (e.g. localhost) as `-master_addr`, which improves the ease of use of msrun.
- [STABLE] msrun supports printing training logs to standard output. Users can control which ranks to print with the `-tail_worker_log` parameter.
- [STABLE] After setting `export VLOG_v=12500`, the `scheduler` log can output cluster information, which helps users to quickly count cluster data.
- [STABLE] msrun supports formatting the log file name with the `--worker_log_name` parameter to help users quickly locate the problem node.

For details, refer to [msrun Launching](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).

#### Profiler

- [STABLE] New interfaces [mindspore.profiler.schedule](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.profiler.schedule.html) and [mindspore.profiler.tensor_board_trace_handler](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.profiler.tensor_board_trace_handler.html) are added to support the acquisition and rendering of PyNative scenarios by step, which improves the ease of use of PyNative scenarios.
- [STABLE] Dynamic Profiling supports customized for loops to enhance the ease of use of dynamic graphical scenarios.
- [STABLE] Profiler initialization parameters and deliverables directory structure aligned to PTA to reduce user migration difficulty.
- [STABLE] A new lightweight interface, [mindspore.profiler.mstx](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.profiler.mstx.html), has been added to provide users with a low-overhead performance data collection method.
- [STABLE] Timeline supports displaying hardware utilization data to help users locate downclocking issues.

For details, refer to [Ascend Performance Tuning](https://www.mindspore.cn/docs/en/r2.5.0/model_train/optimize/profiler.html).

#### PyNative

- [Beta] PyNative mode supports the inplace operator process, introduces the inplace operators. Taking the [mindspore.mint.nn.functional.relu](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mint/mindspore.mint.nn.functional.relu.html) as an example, if you want to use the inplace updated version of the relu operator, you can call the [mindspore.mint.nn.functional.relu_](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mint/mindspore.mint.nn.functional.relu_.html).
- [STABLE] Enable the environment variable MS_SIMULATION_LEVEL=1 to enable the PyNative dryrun, the multi-device process can be simulated without occupying the device, and the display memory usage can be viewed in logs. For details, refer to [Environment Variables](https://www.mindspore.cn/docs/en/r2.5.0/api_python/env_var_list.html?highlight=ms_simulation_level#%E5%88%86%E5%B8%83%E5%BC%8F%E5%B9%B6%E8%A1%8C)。

#### FrontEnd

- [STABLE] Added [mindspore.nn.utils.no_init_parameters](https://www.mindspore.cn/docs/en/r2.5.0/api_python/nn/mindspore.nn.utils.no_init_parameters.html) API, which supports delayed initialization of network parameters and reduces model startup time in inference scenarios.

### API Change

#### New APIs

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore.mint.html) API provides more functional, nn interfaces. The mint interface is currently an experimental interface and performs better than ops in O0/O1 and pynative mode. Currently, the O2 compilation mode (graph sinking mode) and CPU/GPU backend are not supported, and it will be gradually improved in the future.

  | mindspore.mint              |                           |                                   |                            |
  | :-------------------------- | :------------------------ | :-------------------------------- | :------------------------- |
  | mindspore.mint.bernoulli    | mindspore.mint.bincount   | mindspore.mint.clone              | mindspore.mint.einsum      |
  | mindspore.mint.empty        | mindspore.mint.empty_like | mindspore.mint.full_like          | mindspore.mint.randint     |
  | mindspore.mint.randint_like | mindspore.mint.randn      | mindspore.mint.randn_like         | mindspore.mint.randperm    |
  | mindspore.mint.chunk        | mindspore.mint.concat     | mindspore.mint.count_nonzero      | mindspore.mint.scatter     |
  | mindspore.mint.select       | mindspore.mint.squeeze    | mindspore.mint.swapaxes           | mindspore.mint.transpose   |
  | mindspore.mint.triu         | mindspore.mint.unbind     | mindspore.mint.unique_consecutive | mindspore.mint.multinomial |
  | mindspore.mint.addmv        | mindspore.mint.diff       | mindspore.mint.exp2               | mindspore.mint.float_power |
  | mindspore.mint.fix          | mindspore.mint.fmod       | mindspore.mint.frac               | mindspore.mint.lerp        |
  | mindspore.mint.log2         | mindspore.mint.log10      | mindspore.mint.logaddexp          | mindspore.mint.mv          |
  | mindspore.mint.nansum       | mindspore.mint.nan_to_num | mindspore.mint.polar              | mindspore.mint.ravel       |
  | mindspore.mint.outer        | mindspore.mint.softmax    | mindspore.mint.t                  | mindspore.mint.cdist       |
  | mindspore.mint.amax         | mindspore.mint.amin       | mindspore.mint.cumprod            | mindspore.mint.histc       |
  | mindspore.mint.logsumexp    | mindspore.mint.norm       | mindspore.mint.std                | mindspore.mint.std_mean    |
  | mindspore.mint.var          | mindspore.mint.var_mean   | mindspore.mint.allclose           | mindspore.mint.argsort     |
  | mindspore.mint.equal        | mindspore.mint.isinf      | mindspore.mint.isneginf           | mindspore.mint.not_equal   |
  | mindspore.mint.addbmm       | mindspore.mint.addmm      | mindspore.mint.baddbmm            | mindspore.mint.dot         |
  | mindspore.mint.meshgrid     | mindspore.mint.mm         |             |                            |

  | mindspore.mint.nn                   |                                    |
  | :---------------------------------- | ---------------------------------- |
  | mindspore.mint.nn.Conv3d            | mindspore.mint.nn.ConstantPad1d    |
  | mindspore.mint.nn.ConvTranspose2d   | mindspore.mint.nn.ConstantPad2d    |
  | mindspore.mint.nn.BatchNorm1d       | mindspore.mint.nn.ConstantPad3d    |
  | mindspore.mint.nn.BatchNorm2d       | mindspore.mint.nn.ReflectionPad1d  |
  | mindspore.mint.nn.BatchNorm3d       | mindspore.mint.nn.ReflectionPad2d  |
  | mindspore.mint.nn.LayerNorm         | mindspore.mint.nn.ReflectionPad3d  |
  | mindspore.mint.nn.SyncBatchNorm     | mindspore.mint.nn.ReplicationPad1d |
  | mindspore.mint.nn.ELU               | mindspore.mint.nn.ZeroPad1d        |
  | mindspore.mint.nn.GELU              | mindspore.mint.nn.ZeroPad2d        |
  | mindspore.mint.nn.LogSigmoid        | mindspore.mint.nn.ZeroPad3d        |
  | mindspore.mint.nn.ReLU6             | mindspore.mint.nn.BCELoss          |
  | mindspore.mint.nn.SiLU              | mindspore.mint.nn.CrossEntropyLoss |
  | mindspore.mint.nn.Tanh              | mindspore.mint.nn.NLLLoss          |
  | mindspore.mint.nn.Embedding         | mindspore.mint.nn.SmoothL1Loss     |
  | mindspore.mint.nn.Dropout2d         | mindspore.mint.nn.Upsample         |
  | mindspore.mint.nn.AdaptiveAvgPool1d | mindspore.mint.nn.MaxUnpool2d      |
  | mindspore.mint.nn.AdaptiveAvgPool2d |                                    |

  | mindspore.mint.nn.functional                     |
  | :----------------------------------------------- |
  | mindspore.mint.nn.functional.adaptive_avg_pool1d |
  | mindspore.mint.nn.functional.adaptive_avg_pool2d |
  | mindspore.mint.nn.functional.avg_pool1d          |
  | mindspore.mint.nn.functional.max_unpool2d        |
  | mindspore.mint.nn.functional.logsigmoid          |
  | mindspore.mint.nn.functional.relu6               |
  | mindspore.mint.nn.functional.relu_               |
  | mindspore.mint.nn.functional.normalize           |
  | mindspore.mint.nn.functional.dropout2d           |
  | mindspore.mint.nn.functional.nll_loss            |
  | mindspore.mint.nn.functional.smooth_l1_loss      |
  | mindspore.mint.nn.functional.interpolate         |
  | mindspore.mint.nn.functional.conv3d              |

  | mindspore.mint.distributed                        |                                                    |
  | ------------------------------------------------- | -------------------------------------------------- |
  | mindspore.mint.distributed.all_gather             | mindspore.mint.distributed.get_global_rank         |
  | mindspore.mint.distributed.all_gather_into_tensor | mindspore.mint.distributed.get_group_rank          |
  | mindspore.mint.distributed.all_gather_object      | mindspore.mint.distributed.get_process_group_ranks |
  | mindspore.mint.distributed.all_reduce             | mindspore.mint.distributed.init_process_group      |
  | mindspore.mint.distributed.all_to_all             | mindspore.mint.distributed.irecv                   |
  | mindspore.mint.distributed.all_to_all_single      | mindspore.mint.distributed.isend                   |
  | mindspore.mint.distributed.barrier                | mindspore.mint.distributed.new_group               |
  | mindspore.mint.distributed.batch_isend_irecv      | mindspore.mint.distributed.P2POp                   |
  | mindspore.mint.distributed.broadcast              | mindspore.mint.distributed.recv                    |
  | mindspore.mint.distributed.broadcast_object_list  | mindspore.mint.distributed.reduce                  |
  | mindspore.mint.distributed.gather                 | mindspore.mint.distributed.reduce_scatter          |
  | mindspore.mint.distributed.gather_object          | mindspore.mint.distributed.reduce_scatter_tensor   |
  | mindspore.mint.distributed.get_backend            | mindspore.mint.distributed.scatter                 |
  | mindspore.mint.distributed.scatter_object_list    | mindspore.mint.distributed.send                    |

  | others                            |
  | --------------------------------- |
  | mindspore.mint.optim.Adam         |
  | mindspore.mint.linalg.matrix_norm |
  | mindspore.mint.linalg.norm        |
  | mindspore.mint.linalg.vector_norm |
  | mindspore.mint.special.exp2       |

- [STABLE] Two inference API [mindspore.ops.incre_flash_attention](https://www.mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.incre_flash_attention.html) and [mindspore.ops.prompt_flash_attention](https://www.mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.prompt_flash_attention.html) are added. Currently, only Ascend backend is supported.
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore.runtime.html) replaces the original mindspore.hal interfaces and provides interfaces related to runtime resources such as stream, memory, and event.
- [STABLE] [mindspore.device_context](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore.device_context.html) replaces some parameters of the original set_context interface and provides setting interfaces related to hardware platform.
- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API provides more Tensor methods. Currently the Tensor interfaces are still the experimental interfaces, and do not support the graph sink mode and CPU, GPU back-end, will be gradually improved. In addition, a large number of existing Tensor methods, including operators like +=, -=, *= and /=, have been adapted with Aclnn kernels on the Ascend backend through overloading. Details can be found in [API list](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) in official website.

#### Backwards Incompatible Change

- For API [mindspore.Tensor.new_ones](https://gitee.com/link?target=https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/Tensor/mindspore.Tensor.new_zeros.html), the input argument "size" no longer supports the data type of Tensor.
- mindspore.Profiler removes timeline_limit, rank_id, analyze_only, env_enable parameters.

- Interface name: mindspore.Profiler

  Changes: profile_communication is deprecated and the communication matrix data is collected by setting profiler_level=ProfilerLevel.Level1 or profiler_level=ProfilerLevel.Level2.

  Clarification: The default value of profiler_level is ProfilerLevel.Level0.

  <table>
  <tr>
  <td style="text-align:center"> original interface  </td> <td style="text-align:center"> v2.5.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  Profiler(profile_communication=True)
  </pre>
  </td>
  <td><pre>
  Profiler(profiler_level=ProfilerLevel.Level1) or Profiler(profiler_level=ProfilerLevel.Level2)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.Profiler

  Changes: op_time is deprecated, set to collect NPU-side operator performance data by setting activaties=[mindspore.profiler.ProfilerActivity.NPU].

  Clarification: The activaties parameter is of type list, and as long as it contains the mindspore.profiler.ProfilerActivity.NPU parameter, it means that the collection of NPU-side operator performance data is enabled, and the collection is turned on by default.

  <table>
  <tr>
  <td style="text-align:center"> original interface </td> <td style="text-align:center"> v2.5.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  Profiler(op_time=True)
  </pre>
  </td>
  <td><pre>
  Profiler(activaties=[mindspore.profiler.ProfilerActivity.NPU])
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.Profiler

  Changes: The type of aicore_metrics changed from int to mindspore.profiler.AicoreMetrics enum value.

  Clarification: The default value of aicore_metrics is mindspore.profiler.AicoreMetric.AiCoreNone.

  <table>
  <tr>
  <td style="text-align:center"> original interface </td> <td style="text-align:center"> v2.5.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  Profiler(aicore_metrics=0)
  </pre>
  </td>
  <td><pre>
  Profiler(aicore_metrics=mindspore.profiler.AicoreMetric.AiCoreNone)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.Profiler

  Changes: profile_framework is deprecated and the frame measurements are collected by setting activaties=[mindspore.profiler.ProfilerActivity.CPU].

  Clarification: activaties parameter type is list, as long as it contains mindspore.profiler.ProfilerActivity.CPU parameter means enable to collect the framework performance data, the default on the collection.

  <table>
  <tr>
  <td style="text-align:center"> original interface </td> <td style="text-align:center"> v2.5.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  Profiler(profile_framework="all")
  </pre>
  </td>
  <td><pre>
  Profiler(activaties=[mindspore.profiler.ProfilerActivity.CPU])
  </pre>
  </td>
  </tr>
  </table>

### Contributors

baishanyang ,bantao ,Bellatan ,biangelin ,BigSkySea ,caifubi ,candanzg ,candyhong ,Carey ,cccc1111 ,chaijinwei ,changzherui ,chengbin ,chengfeng27 ,chengxb7532 ,chujinjin ,coder2237 ,czrz ,dairenjie ,DavidFFFan ,DeshiChen ,dingjinshan ,ehaleva ,Erpim ,fary86 ,fengyixing ,ffmh ,fuchao ,fuhouyu ,gaoyong10 ,geyuhong ,guoyuzhe ,GuoZhibin ,guozhijian ,halo ,hangq ,haozhang ,hedongdong ,hehongzhe ,hhz886 ,HighCloud ,huangbingjian ,HuangLe02 ,huangziling ,huda ,Huilan Li ,hujiahui8 ,jiahaochen666 ,jiangchao_j ,jiangchenglin3 ,jiangshanfeng ,jiaorui ,jiaxueyu ,jizewei ,jjfeing ,JoeyLin ,jshawjc ,kakyo82 ,kingxian ,kisnwang ,leida ,liangchenghui ,lianghongrui ,LiangZhibo ,lichen ,limingqi107 ,LINH ,linux ,lionelchang ,lishanni ,liubuyu ,liujunzhu ,liuluobin ,liuxu ,liuyanwei ,liyan2022 ,LLLRT ,looop5 ,luochao60 ,luoxuewei ,luoyang ,lyk ,machenggui ,maoyuanpeng1 ,Margaret_wangrui ,master,mengxian ,MengXiangyu ,mengyuanli ,Mrtutu ,mylinchi ,NaCN ,Nikanuo ,niujunhao ,panzhihui ,pengqi ,PingqiLi ,pipecat ,qiuleilei ,qiuyufeng ,qiuzhongya ,r1chardf1d0 ,shaoshengqi ,shen_haochen ,shenhaojing ,shenwei41 ,shilishan ,shiro-zzz ,shuqian0 ,St.Universe ,stavewu ,superxf ,suteng ,TAJh ,tanghuikang ,tangmengcheng ,tan-wei-cheng ,tianxiaodong ,TuDouNi ,TYWZ22259 ,user_0145 ,VectorSL ,vincen45 ,wang_ziqi ,wangshaocong ,wangwensheng4 ,weiyang ,wtcheng ,wtobill ,wujiangming ,wujueying ,wuweikang ,wwwbby ,XianglongZeng ,xiaopeng ,xiaotianci ,xiaoyao ,xiedejin1 ,XinDu ,xuxinglei ,yang guodong ,yangben ,yanghaoran ,yanglong ,yanx ,Yanzhi_YI ,yao_yf ,yefeng ,Yi_zhang95 ,yide12 ,yihangchen ,YijieChen ,YingtongHu ,ylw ,yonibaehr ,yuanqi ,yuchaojie ,yuezenglin ,YuJianfeng ,yyuse ,Zhang QR ,zhangbuxue ,zhangdanyang ,zhanghaibo ,zhangminli ,zhangyinxia ,ZhangZGC ,zhangzhen ,zhengzuohe ,zhouyaqiang0 ,zhuguodong ,zichun_ye ,zlq2020 ,zong_shuai ,ZPaC ,zyli2020 ,陈一 ,程超 ,冯一航 ,胡彬 ,宦晓玲 ,黄勇 ,简云超 ,康伟 ,李栋 ,李良灿 ,李林杰 ,李寅杰,刘崇鸣 ,刘力力 ,刘思铭 ,刘涛Liu ,刘勇琪 ,刘子涵 ,吕浩宇 ,吕凯盟 ,梅飞要 ,倪轩 ,任新 ,十一雷 ,孙昊辰 ,王禹程 ,王振邦 ,熊攀 ,俞涵 ,虞良斌 ,张栩浩 ,赵文璇 ,周莉莉 ,周一航 ,邹文祥

## MindSpore 2.4.1 Release Notes

### Major Features and Improvements

#### AutoParallel

- [STABLE] Split/concat branch communication computation parallel is supported. Users split input data to form parallelizable branches. Automatic communication computing parallelism is performed between branches, reducing communication overhead.
- [STABLE] Sequence pipelines are supported. The LLama series models for the dev branch of MindFormers reduces the Bubble as well as the memory overhead of pipeline parallelism by introducing Sequence dimension splitting.

#### PyNative

- [STABLE] In PyNative mode, communication operators are assigned streams by default based on the communication domain. They support concurrent execution of communication operators, optimize collaborative parallel strategies, provide fine-grained communication masking, and enhance model performance.

### Bug Fixes

- [IB0R4N](https://gitee.com/mindspore/mindspore/issues/IB0R4N): Fixed the problem of loading distributed weights with inaccurate accuracy under certain splitting strategies.

### Contributors

bantao;caifubi;candanzg;chaijinwei;changzherui;chengbin;chujinjin;DeshiChen;dingjinshan;fary86;fuhouyu;gaoyong10;GuoZhibin;halo;haozhang;hedongdong;huangbingjian;hujiahui8;huoxinyou;jiangshanfeng;jiaorui;jiaxueyu;jshawjc;kisnwang;lichen;limingqi107;liubuyu;looop5;luochao60;luoyang;machenggui;MengXiangyu;Mrtutu;NaCN;panzhihui;qiuzhongya;shenhaojing;shilishan;tanghuikang;TuDouNi;wang_ziqi;weiyang;wujueying;XianglongZeng;xuxinglei;yang guodong;yanghaoran;yao_yf;yide12;yihangchen;YijieChen;YingtongHu;yuchaojie;YuJianfeng;zhangdanyang;ZhangZGC;zhengzuohe;zong_shuai;ZPaC;冯一航;胡彬;宦晓玲;李林杰;刘崇鸣;刘勇琪;任新;王禹程;王振邦;熊攀;俞涵;张栩浩;周一航;

## MindSpore 2.4.0 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] Modify the default value of the `max_rowsize` parameter in the interface [mindspore.dataset.GeneratorDataset](https://www.mindspore.cn/docs/en/r2.4.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html), [mindspore.dataset.Dataset.map](https://www.mindspore.cn/docs/en/r2.4.0/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.map.html), and [mindspore.dataset.Dataset.batch](https://www.mindspore.cn/docs/en/r2.4.0/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html) to None to enable dynamic allocation of shared memory by default, in which case the shared memory will be requested in real time with the input data and accelerate the data processing, so the user does not need to adjust the size of this parameter in advance.
- [BETA] Data processing supports the independent process mode, which will reduce the GIL lock conflict between the training process and the data reading process to improve the performance in dynamic graph mode. This mode can be enabled or disabled via the environment variable `MS_INDEPENDENT_DATASET`.

#### Ascend

- [STABLE] Customized operators support the Ascend Dynamics graph scenario Pyboost execution mode, which reduces operator call overhead.
- [STABLE] The Ascend Print operator supports scenarios where the output is oversized tensor or print calls are intensive, and users can specify the slice size and timeout time to support different scenarios via the `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME` environment variables.
- [STABLE] Unified deterministic computation settings. Users can enable ascending deterministic computation by only setting `mindspore.set_context(deterministic="ON")`.
- [STABLE] Supports aggregate communication anomaly monitoring, quickly exits training after monitoring communication anomalies to avoid timeout waiting.
- [STABLE] Supports the [graceful exit function for sub-healthy devices](https://www.mindspore.cn/docs/en/r2.4.0/model_train/train_availability/graceful_exit.html). When the training framework detects the presence of sub-healthy device configuration information in the cluster, it saves the CKPT and uniformly ends the cluster training process.

#### Runtime

- [STABLE] Backend compilation cache is supported in O0/O1 mode and is turned on by default when frontend compilation cache is turned on.
- [STABLE] The aclnnAllGatherMatmul, aclnnMatmulReduceScatter, and aclnnMatmulAllReduce algorithms are supported in O0/O1 modes to improve performance.
- [STABLE] O0/O1 modes support to disable cluster heartbeat configuration by export MS_DISABLE_HEARTBEAT=1 to reduce scheduler load.
- [STABLE] O0/O1 modes support communication arithmetic fusion.
- [STABLE] Virtual memory support in O2 mode, defragmentation support, which is enabled in Ascend backend by default.
- [STABLE] Dynamic request for device memory occupation, support single card multi-user use, which is enabled in Ascend backend by default.
- [STABLE] Optimize graph fusion compilation performance in O1 mode, enabled by default.
- [STABLE] Support kernel packet fusion optimization in O1 mode to improve the performance of dynamic shape network execution, enabled by default.
- [BETA] Epilogue fusion between the MatMul and Elementwise operator is supported in O1 mode. Enable via `mindspore.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")`.
- [BETA] O1 mode supports user-controlled graph fusion optimization scope, user can control to turn on or off the corresponding fusion operator via the enable_pass/disable_pass option of graph_kernel_flags.

#### PyNative

- [STABLE] Parameter cell_id of Hook function corresponding to [mindspore.nn.Cell.register_backward_hook](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook) and [mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook) is changed to cell's python object.
- [STABLE] Added [Cell.register_backward_pre_hook](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_pre_hook) interface, this API registers the backward propagation hook function on a Cell, which is called each time the gradient of that Cell is computed.
- [STABLE] Optimize the PyNative process AICPU class operator downstream cache to improve API execution performance.
- [STABLE] Added the function of converting the device memory occupied by a group of Tensor to a contiguous piece of memory under dynamic graph.

#### FrontEnd

- [STABLE] Weight de-redundancy saving and loading is supported in fault recovery scenarios.
- [STABLE] Mixed precision training with support for [auto mode](https://www.mindspore.cn/docs/en/r2.4.0/api_python/amp/mindspore.amp.auto_mixed_precision.html#mindspore.amp.auto_mixed_precision).
- [STABLE] Support saving and loading of safetensors format, as well as offline aggregation and distributed loading based on safetensors in parallel scenarios.
- [BETA] Added new cyclic arithmetic interface [mindspore.ops.WhileLoop](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.WhileLoop.html), [mindspore.ops.ForiLoop](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.ForiLoop.html), and [mindspore.ops.Scan](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.Scan.html), optimizing loop compilation time.
- [BETA] The graph mode supports the operator passing keyword arguments.

#### Parallel

- [STABLE] [mindspore.ops.TensorDump](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.TensorDump.html) operator supports distributed parallel scenarios, and users can decide to print input/output slices by configuring the TensorDump operator's `input_output` attribute; add new interface [mindspore.ops.tensordump](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.tensordump.html).
- [STABLE] msrun supports customizing the rank id based on the passing rank table file, and supports rearranging the rank id via the `--rank_table_file` passing json file.
- [STABLE] Supports LCCL, a high-performance communication library in Ascend stand-alone. Users can enable LCCL in Ascend back-end training scenarios via the `MS_ENABLE_LCCL` environment variable.
- [STABLE] The strategy propagation algorithm is adapted to LLaMA/Mixtral networks, which reduces the workload of users in configuring the sharding strategy for LLaMA/Mixtral networks.
- [STABLE] Support high dimensional tensor parallelism, user can configure [mindspore.ops.MatMul](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.MatMul.html) and [mindspore.oops.BatchMatMul](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.BatchMatMul.html) input_layout switching 1D/2D/3D tensor slice mode.
- [STABLE] Simulation compilation does not consume hardware resources when SIMULATION_LEVEL=0 and SIMULATION_LEVEL=1 runtime jit_level is O0/O1.
- [STABLE] Allreduce introduced in parallel by the BatchMatMul model is automatically converted to a ReduceScatter to reduce communication according to the matching rules if enable_allreduce_slice_to_reducescatter is turned on in parallel_speed_up_json when it follows the slice operation.
- [STABLE] [mindspore.nn.Cell.shard](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.shard) and [mindspore.shard](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore/mindspore.shard.html) support user-configurable policies of type mindspore.Layout and sharding strategy for each parameter parameter_plan.
- [BETA] SAPP supports fully automatic generation of residual arithmetic policies after manual preconfiguration of arithmetic parallel sharding strategy. The user activates the `.shard()` preconfigured parallel sharding strategy by turning on the `MS_INTERFERED_SAPP` environment variable.
- The [BETA] [mindspore.ops.Custom](https://www.mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.Custom.html) operator supports configuring the sharding strategy.

#### Inference

- [STABLE] New Qwen2 and LLaMA3.1 series of large models support training and inference architecture, realize the unification of script, distributed policy and runtime, reduce the inference delay by fusing large operators, and effectively improve the network throughput.
- [STABLE] Support parallel decoding service-oriented deployment to realize LookAhead speculative inference for large models of LLaMA series.
- [BETA] Support SLoRA service-oriented deployment, realizing multi-trimming weight scheduling inference for large models.

#### Dump

- [STABLE] Optimize [Dump](https://www.mindspore.cn/docs/en/r2.4.0/model_train/debug/dump.html) for use by device type and optimization level.
- [STABLE] Asynchronous Dump support in Ascend O0/O1 mode, including asynchronous Tensor, overflow, and statistics (host and device modes).
- [STABLE] Overflow Dump supports configuring the maximum number of overflows.
- [STABLE] Ascend O2 mode supports set dump.
- [STABLE] Support qint4 x 2 quantization type Dump.

### API Change

#### New API

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore.mint.html) APIs add a large number of functional, nn interfaces. mint interfaces are currently experimental interfaces, performance is better than ops in graph compilation mode O0 and PyNative mode. Currently does not support graph sink mode and CPU, GPU backend. It will be gradually improved.

  | mindspore.mint             |                                  |                              |                            |
  | :------------------------- | :------------------------------- | :--------------------------- | :------------------------- |
  | mindspore.mint.full        | mindspore.mint.repeat_interleave | mindspore.mint.linspace      | mindspore.mint.scatter     |
  | mindspore.mint.tril        | mindspore.mint.argmin            | mindspore.mint.sign          | mindspore.mint.remainder   |
  | mindspore.mint.flatten     | mindspore.mint.asin              | mindspore.mint.arcsin        | mindspore.mint.sinh        |
  | mindspore.mint.arcsinh     | mindspore.mint.atan              | mindspore.mint.arctan        | mindspore.mint.atanh       |
  | mindspore.mint.arctanh     | mindspore.mint.acos              | mindspore.mint.arccos        | mindspore.mint.acosh       |
  | mindspore.mint.arccosh     | mindspore.mint.erfc              | mindspore.mint.expm1         | mindspore.mint.log1p       |
  | mindspore.mint.logical_xor | mindspore.mint.round             | mindspore.mint.tan           | mindspore.mint.trace       |
  | mindspore.mint.trunc       | mindspore.mint.cross             | mindspore.mint.masked_select | mindspore.mint.bitwise_and |
  | mindspore.mint.bitwise_or  | mindspore.mint.bitwise_xor       | mindspore.mint.cosh          | mindspore.mint.cummax      |
  | mindspore.mint.cummin      | mindspore.mint.median            | mindspore.mint.roll          | mindspore.mint.sinc        |
  | mindspore.mint.sinh        | mindspore.mint.xlogy             |                              |                            |

  | mindspore.mint.nn             |
  | :---------------------------- |
  | mindspore.mint.nn.ReLU        |
  | mindspore.mint.nn.Hardsigmoid |
  | mindspore.mint.nn.AvgPool2d   |
  | mindspore.mint.nn.MSELoss     |
  | mindspore.mint.nn.LogSoftmax  |
  | mindspore.mint.nn.Mish        |
  | mindspore.mint.nn.PReLU       |
  | mindspore.mint.nn.SELU        |
  | mindspore.mint.nn.Softshrink  |
  | mindspore.mint.nn.Hardshrink  |
  | mindspore.mint.nn.Hardswish   |
  | mindspore.mint.nn.L1Loss      |

  | mindspore.mint.nn.functional             |
  | :--------------------------------------- |
  | mindspore.mint.nn.functional.hardsigmoid |
  | mindspore.mint.nn.functional.log_softmax |
  | mindspore.mint.nn.functional.mish        |
  | mindspore.mint.nn.functional.prelu       |
  | mindspore.mint.nn.functional.selu        |
  | mindspore.mint.nn.functional.softshrink  |
  | mindspore.mint.nn.functional.hardshrink  |
  | mindspore.mint.nn.functional.hardswish   |
  | mindspore.mint.nn.functional.l1_loss     |
  |                                          |

#### Interface Changes

- Interface name: mindspore.dataset.GeneratorDataset

  Changed: The default value of parameter `max_rowsize` is changed from `6` to `None` to enable dynamic allocation of shared memory by default.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  class GeneratorDataset(source,
                         column_names=None,
                         column_types=None,
                         schema=None,
                         num_samples=None,
                         num_parallel_workers=1,
                         shuffle=None,
                         sampler=None,
                         num_shards=None,
                         shard_id=None,
                         python_multiprocessing=True,
                         max_rowsize=6)
  </pre>
  </td>
  <td><pre>
  class GeneratorDataset(source,
                         column_names=None,
                         column_types=None,
                         schema=None,
                         num_samples=None,
                         num_parallel_workers=1,
                         shuffle=None,
                         sampler=None,
                         num_shards=None,
                         shard_id=None,
                         python_multiprocessing=True,
                         max_rowsize=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.dataset.Dataset.batch

  Changed: The default value of parameter `max_rowsize` is changed from `16` to `None` to enable dynamic allocation of shared memory by default.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def batch(input_dataset,
            batch_size,
            drop_remainder=False,
            num_parallel_workers=None,
            per_batch_map=None,
            input_columns=None,
            output_columns=None,
            python_multiprocessing=False,
            max_rowsize=16)
  </pre>
  </td>
  <td><pre>
  def batch(input_dataset,
            batch_size,
            drop_remainder=False,
            num_parallel_workers=None,
            per_batch_map=None,
            input_columns=None,
            output_columns=None,
            python_multiprocessing=False,
            max_rowsize=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.dataset.Dataset.map

  Changed: The default value of parameter `max_rowsize` is changed from `16` to `None` to enable dynamic allocation of shared memory by default.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def map(input_dataset,
          operations=None,
          input_columns=None,
          output_columns=None,
          num_parallel_workers=None,
          python_multiprocessing=False,
          cache=None,
          callbacks=None,
          max_rowsize=16, offload=None)
  </pre>
  </td>
  <td><pre>
  def map(input_dataset,
          operations=None,
          input_columns=None,
          output_columns=None,
          num_parallel_workers=None,
          python_multiprocessing=False,
          cache=None,
          callbacks=None,
          max_rowsize=None, offload=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.ops.TensorDump

  Changed: New parameter `input_output` to control printing behavior.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  class TensorDump()
  </pre>
  </td>
  <td><pre>
  class TensorDump(input_output='out')
  </pre>
  </td>
  </tr>
  </table>

- Interface name: File formats saved by MindSpore Dump Tensor

  Changed: The npy file obtained by Dump adds the dtype information of the original Tensor to the filename.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.
  {format}.npy
  </pre>
  </td>
  <td><pre>
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.
  {format}.{dtype}.npy
  </pre>
  </td>
  </tr>
  </table>

#### Non-compatible Interface Changes

- Interface name: mindspore.nn.Cell.register_backward_hook(hook_fn)

  Changed: The input parameter of hook_fn is changed from cell_id to cell object.

  Descriptions: For the original hook, you can get the original cell_id by id(cell) in hook_fn.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def register_backward_hook(hook_fn)
  Parameter hook_fn(cell_id,
             grad_input, grad_output)
             -> New grad_output or None
  </pre>
  </td>
  <td><pre>
  def register_backward_hook(hook_fn)
  Parameter hook_fn(cell,
             grad_input, grad_output)
             -> New grad_input or None
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.nn.Cell.register_forward_hook(hook_fn)

  Changed: The input parameter of hook_fn is changed from cell_id to cell object.

  Descriptions: For the original hook, you can get the original cell_id by id(cell) in hook_fn.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def register_forward_hook(hook_fn)
  Parameter hook_fn(cell_id, inputs, outputs)-> New outputs or None
  </pre>
  </td>
  <td><pre>
  def register_forward_hook(hook_fn)
  Parameter hook_fn(cell, inputs, outputs)-> New outputs or None
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.all_reduce

  Changed: all_reduce adds a new parameter async_op, and the return value is changed from Tensor to a tuple consisting of Tensor and CommHandle.

  Descriptions: async_op indicates whether all_reduce has multi-stream parallelism turned on, and the default value is False.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def all_reduce(tensor,
                 op=ReduceOp.SUM,
                 group=GlobalComm.WORLD_COMM_GROUP)->Tensor
  </pre>
  </td>
  <td><pre>
  def all_reduce(tensor,
                 op=ReduceOp.SUM,
                 group=GlobalComm.WORLD_COMM_GROUP,
                 async_op=False)
                 ->tuple(Tensor, CommHandle)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.all_gather_into_tensor

  Changed: all_reduce adds a new parameter async_op, and the return value is changed from Tensor to a tuple consisting of Tensor and CommHandle.

  Descriptions: async_op indicates whether all_gather_into_tensor has multi-stream parallelism turned on, and the default value is False.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def all_gather_into_tensor(tensor,
                             group=GlobalComm.
                             WORLD_COMM_GROUP)->Tensor
  </pre>
  </td>
  <td><pre>
  def all_gather_into_tensor(tensor,
                             group=GlobalComm.
                             WORLD_COMM_GROUP,
                             async_op=False)->
                             tuple(Tensor, CommHandle)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.reduce_scatter_tensor

  Changed: all_reduce adds a new parameter async_op, and the return value is changed from Tensor to a tuple consisting of Tensor and CommHandle.

  Descriptions: async_op indicates whether reduce_scatter_tensor has multi-stream parallelism turned on, and the default value is False.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def reduce_scatter_tensor(tensor,
                            op=ReduceOp.SUM,
                            group=GlobalComm.
                            WORLD_COMM_GROUP)->Tensor
  </pre>
  </td>
  <td><pre>
  def reduce_scatter_tensor(tensor,
                            op=ReduceOp.SUM,
                            group=GlobalComm.WORLD_COMM_GROUP,
                            async_op=False)->
                            tuple(Tensor, CommHandle)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.isend

  Changed: The return value is changed from Tensor to Handle.

  Descriptions: isend enables multi-stream parallelism by default.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def isend(tensor,
            dst=0,group=GlobalComm.
            WORLD_COMM_GROUP, tag=0)->Tensor
  </pre>
  </td>
  <td><pre>
  def isend(tensor,
            dst=0,group=GlobalComm.
            WORLD_COMM_GROUP, tag=0)->CommHandle
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.irecv

  Changed: The return value is changed from Tensor to Handle.

  Descriptions: irecv enables multi-stream parallelism by default.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def irecv(tensor,
            src=0, group=GlobalComm.
            WORLD_COMM_GROUP, tag=0)->Tensor
  </pre>
  </td>
  <td><pre>
  def irecv(tensor,
            src=0,
            group=GlobalComm.
            WORLD_COMM_GROUP, tag=0)->CommHandle
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.all_to_all_with_output_shape

  Changed: all_to_all_with_output_shape adds a new parameter async_op, and the return value is changed from Tensor to a tuple consisting of Tensor and CommHandle.

  Descriptions: async_op indicates whether all_to_all_with_output_shape enables multi-stream parallelism, the default value is False.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def all_to_all_with_output_shape(output_shape_list,
                                   input_tensor_list,
                                   group=None)->tuple(Tensor)
  </pre>
  </td>
  <td><pre>
  def all_to_all_with_output_shape(output_shape_list,
                                   input_tensor_list,
                                   group=None,
                                   async_op=False)->
                                   tuple(tuple(Tensor),
                                   CommHandle)
  </pre>
  </td>
  </tr>
  </table>

- Interface name: mindspore.communication.comm_func.all_to_all_single_with_output_shape

  Changed: all_to_all_single_with_output_shape adds a new parameter async_op, and the return value is changed from Tensor to a tuple consisting of Tensor and CommHandle.

  Descriptions: async_op indicates whether all_to_all_single_with_output_shape enables multi-stream parallelism, the default value is False.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.4.0 interface </td>
  </tr>
  <tr>
  <td><pre>
  def all_to_all_single_with_output_shape(output_shape,
                                          tensor,
                                          output_split_sizes=None,
                                          input_split_sizes=None,
                                          group=None)->Tensor
  </pre>
  </td>
  <td><pre>
  def all_to_all_single_with_output_shape(output_shape,
                                          tensor,
                                          output_split_sizes=None,
                                          input_split_sizes=None,
                                          group=None,
                                          async_op=False)->
                                          tuple(Tensor, CommHandle)
  </pre>
  </td>
  </tr>
  </table>

### Contributors

anyrenwei,bantao,baochong,Bellatan,BJ-WANG,caifubi,candanzg,candyhong,Carey,cccc1111,ccsszz,changzherui,chengbin,chengfeng27,chengxb7532,chenjianping,chenweifeng,chujinjin,dairenjie,DavidFFFan,DeshiChen,dingjinshan,emmmmtang,fanyi20,fary86,fengyixing,fix-dryrun,fuchao,fuhouyu,gaoyong10,gengdongjie,gent1e,GuoZhibin,guozhijian,halo,hangq,haozhang,hedongdong,Henry Shi,HighCloud,Hongxing,huandong1,huangbingjian,HuangLe02,huangziling,huda,huiliang166,hujiahui8,huoxinyou,jiangchenglin3,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jijiarong,jjfeing,JoeyLin,jshawjc,jxl,kairui_kou,kisnwang,kk,lanzhineng,LiangZhibo,lichen,limingqi107,lionelchang,liubuyu,liujunzhu,liuluobin,liyejun,LLLRT,looop5,luochao60,luoxuewei,luoyang,machenggui,maning202007,maoyuanpeng1,Margaret_wangrui,MengXiangyu,mengyuanli,moran,Mrtutu,mylinchi,NaCN,nomindcarry,panzhihui,paolopoggi,pengqi,pierreleca,qiuleilei,qiuyufeng,qiuzhongya,r1chardf1d0,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shihlCST,shilishan,shiro-zzz,shiziyang,shop-pin,shunyuanhan,shuqian0,stavewu,superxf,suteng,tanghuikang,tangmengcheng,tan-wei-cheng,tan-wei-cheng-3260,tianxiaodong,TronZhang,TuDouNi,VectorSL,vincen45,wang_ziqi,wanghenchang,wangjie,wangshaocong,weiyang,wtobill,wudawei,wujueying,wwwbby,xfan233,XianglongZeng,xiaotianci,xiaoxin_zhang,xiaoxiongzhu,xiaoxuanKL,xiaoyao,XinDu,xuxinglei,xuzhubin,yanghaoran,yanglong,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yefeng,yide12,yihangchen,YijieChen,YingLai Lin,ylw,yuanpeng2024,yuanqi,yuchaojie,Yuheng Wang,YuJianfeng,YukioZzz,yyuse,zangqx,ZeyuHan,zhangbuxue,zhanghaibo,zhangminli,zhangqinghua,zhangyanhui,ZhangZGC,zhangzhen,zhanzhan,zhengzuohe,zhouyaqiang0,zhuguodong,zichun_ye,zjun,zong_shuai,ZPaC,zuochuanyong,zyli2020,程超,蛋蛋de忧桑,狄新凯,范吉斌,冯一航,付国华,胡彬,宦晓玲,黄勇,黄卓,康伟,李良灿,李林杰,李寅杰3,刘崇鸣,刘思铭,刘涛Liu,刘勇琪,刘子涵,吕浩宇,吕昱峰（Nate.River）,钱丹,十一雷,孙昊辰,王禹程,王振邦,王梓润,吴大维,熊攀,徐安越,许子豪,俞涵,云骑士,张峻源,张王泽,张栩浩,赵文璇,周莉莉,朱家兴,邹文祥

## MindSpore 2.3.1 Release Notes

### Major Features and Improvements

- [STABLE] Remove the restriction that the value of device_matrix must be 2 correspongding to interleaved_parallel when using [Layout](https://www.mindspore.cn/docs/en/r2.3.1/api_python/mindspore/mindspore.Layout.html) to construct the parallel strategy.
- [STABLE] Add user-defined control edges environment [MS_CUSTOM_DEPEND_CONFIG_PATH](https://www.mindspore.cn/docs/en/r2.3.1/note/env_var_list.html) support to achieve better overlapping of communication and computation.

### API Change

#### New API

- [STABLE] Add new API [mindspore.mint.repeat_interleave](https://www.mindspore.cn/docs/en/r2.3.1/api_python/mint/mindspore.mint.repeat_interleave.html).

### Contributors

ccsszz;dairenjie;DeshiChen;fuhouyu;gaoshuanglong;gaoyong10;GuoZhibin;halo;huoxinyou;jiangchao_j;jiaorui;jiaxueyu;jijiarong;JuiceZ;lichen;liujunzhu;liuluobin;LLLRT;looop5;luoyang ;Margaret_wangrui;mengyuanli;panzhihui;pengqi;PingqiLi;Renyuan Zhang;tanghuikang;tianxiaodong;TuDouNi;wudawei;XianglongZeng;xiaosh;xiaoxin_zhang;XinDu;yanghaoran;yanglong;yangruoqi713;Yanzhi_YI;yao_yf;YijieChen;yuchaojie;YuJianfeng;zangqx;zhengzuohe;zhouyaqiang0;ZPaC;zyli2020;胡彬;宦晓玲;康伟;李林杰;刘崇鸣;王禹程;俞涵;周莉莉;邹文祥

Contributions of any kind are welcome!

## MindSpore Lite 2.3.1 Release Notes

### Major Features and Improvements

When converting Ascend backend models, the [input_shape](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/cloud_infer/converter_tool_ascend.html) parameter in the configuration file is supported to specify the input size.

### API Change

- [ModelGroup](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/cloud_infer/runtime_cpp.html) interface adds model weight sharing support to save video memory.

- [Model.get_model_info](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/converter_tool.html?highlight=get_model_info) interface adds support for obtaining the input size of the model.

### Contributors

熊攀;ZhangZGC;jxl;zhangyanhui;emmmmtang;huandong1;yefeng

## MindSpore 2.3.0 Release Notes

### Major Features and Improvements

#### AutoParallel

- [STABLE] Extend functional parallelism. [mindspore.shard](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore/mindspore.shard.html) supports now the Graph mode. In Graph mode, the parallel sharding strategy of input and weight can be set for nn.Cell/function. For other operators, the parallel strategy can be automatically configured through "sharding_propagation". Add [mindspore.reshard](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore/mindspore.reshard.html) interface that supports manual rearranging and set up a precise sharding strategy ([mindspore.Layout](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore/mindspore.Layout.html)) for tensors.
- [STABLE] Added Callback interface [mindspore.train.FlopsUtilizationCollector](https://www.mindspore.cn/docs/en/r2.3.0/api_python/train/mindspore.train.FlopsUtilizationCollector.html) statistical model flops utilization information MFU and hardware flops utilization information HFU.
- [STABLE] Add functional communication API [mindspore.communication.comm_func](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore.communication.comm_func.html).
- [BETA] Optimize the memory usage of interleaved pipeline in O0 and O1 mode.
- [BETA] AutoParallel supports automatic pipeline strategy generation in multi-nodes scenarios (not supported in single-node scenario). Need to set `parallel_mode` to ``auto_parallel`` and `search_mode` to ``recursive_programming``.

#### PyNative

- [STABLE] Optimize the basic data structure of PyNative and improve operator API performance.
- [STABLE] Tensor supports [register_hook](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html) so that users can print or modify the gradient with respect to the tensor.
- [STABLE] The PyNative mode supports the recompute function. You can use the recompute interface to reduce the peak device memory of the network.

#### FrontEnd

- [STABLE] Optimize Checkpoint saving and loading basic processes to improve performance by 20%.
- [STABLE] Support CRC verification of Checkpoint files during saving and loading processes to enhance security.

#### Dataset

- [STABLE] Support Ascend processing backend for the following transforms: Equalize, Rotate, AutoContrast, Posterize, AdjustSharpness, Invert, Solarize, ConvertColor, Erase.
- [STABLE] Support video files reading and parsing function. For more detailed information, see APIs: [mindspore.dataset.vision.DecodeVideo](https://www.mindspore.cn/docs/en/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.DecodeVideo.html), [mindspore.dataset.vision.read_video](https://www.mindspore.cn/docs/en/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.read_video.html#mindspore.dataset.vision.read_video), and [mindspore.dataset.vision.read_video_timestamps](https://www.mindspore.cn/docs/en/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.read_video_timestamps.html#mindspore.dataset.vision.read_video_timestamps).
- [STABLE] Support specifying the `max_rowsize` parameter as -1 in `mindspore.dataset.GeneratorDataset`, `mindspore.dataset.Dataset.map` and `mindspore.dataset.Dataset.batch` interfaces. The size of shared memory used by the dataset multiprocessing will be dynamically allocated according to the size of the data. The `max_rowsize` parameter does not need to be adjusted manually.

#### Inference

- [STABLE] 14 large models such as LLaMa2, LLaMa3, and Qwen1.5 are added to support the integrated training and inference architecture to unify scripts, distributed strategies, and runtime. The period from training to inference deployment of typical large models is reduced to days. Large operators are integrated to reduce the inference latency and effectively improve the network throughput.

#### PIJIT

- [BETA] Support bytecode parsing for Python 3.8 and Python 3.10 to expand the supporting version of Python.
- [BETA] Support dynamic shape and symbolic shape as input to enable the dynamic input scenarios.
- [BETA] Enable single-step composition capability to optimize compile time
- [BETA] Support bytecode capture with side effects (STORE_ATTR, STORE_GLOBAL, LIST_APPEND, dict.pop) by bytecode tuning, enabling auto-mixed precision, reduction of cleavage diagrams, and improved performance.

#### Profiler

- [STABLE] Provides a hierarchical Profiler function, controls different levels of performance data collection through the profiler_level parameter.
- [STABLE] Profiler analyse adds a new mode parameter to configure asynchronous parsing mode to parallelize performance data parsing and training.
- [STABLE] The Profiler adds a new data_simplification parameter, which allows users to control whether to delete redundant data after parsing the performance data to save hard disk space.
- [STABLE] The Profiler enhances the memory analysis function. Users can collect the memory application and release information of the framework, CANN and hardware through the profile_memory parameter, and visualize and analyze the information through the [MindStudio tool](https://www.hiascend.com/developer/blog/details/0230130822583032044).
- [BETA] In Pynative mode, Timeline integrates host profiling information, including task time and user side stack information.

#### Dump

- [STABLE] Enhanced synchronous & asynchronous dump functionality and adds L2Norm information to statistics dumps, and the statistic_category field to allow users to customize which statistics to save, improving dump usability. For details about the support for synchronous/asynchronous dump, see [Dump Introduction](https://www.mindspore.cn/tutorials/experts/en/r2.3.0/debug/dump.html#dump-introduction).
- [STABLE] Improved synchronous dump functionality: Enables overflow and exception dumps through the op_debug_mode field.
- [STABLE] Enhanced synchronous dump functionality: The stat_calc_mode field enables device-side computation of statistics (default is host-side), and the sample_mode field is configured to perform sample-based dumps, improving dump performance.
- [STABLE] Enhanced asynchronous dump functionality: Now supports saving in complex64 and complex128 formats.

#### Runtime

- [Stable] Supports multi-level compilation of the staic graph by setting [mindspore.set_context(jit_config={"jit_level": "O0/O1/O2"})](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore/mindspore.set_context.html). The default value is empty, the framework automatically selects the optimization level according to the product category, O2 for Atlas training products and O0 for the rest of the products.
- [Stable] Staic graph supports multi-stream concurrent execution of communication calculations in O0/O1.
- [STABLE] Add memory management API [mindspore.hal.memory](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore.hal.html#memory).
- [Beta] The memory pool supports virtual memory defragmentation, and virtual memory is enabled by default under graph O0/O1.

#### Ascend

- [STABLE] Provide an operator memory out of bounds access detection switch on the Ascend platform, where users can detect internal memory out of bounds issues of operators on the Ascend platform by setting `mindspore.set_context (Ascend_configuration={"op_debug_option": "oom"})`.
- [BETA] The environment variable [MS_SIMULATION_LEVEL](https://www.mindspore.cn/docs/en/r2.3.0/note/env_var_list.html) supports graph compilation O0 execution mode on the Ascend platform, which can support compilation performance and runtime memory analysis
- [BETA] Ascend platform supports [AscendC custom operators](https://www.mindspore.cn/tutorials/experts/en/r2.3.0/operation/op_custom_ascendc.html) through AOT.

### API Change

#### New APIs

- [STABLE] Adds [mindspore.mint](https://www.mindspore.cn/docs/en/r2.3.0/api_python/mindspore.mint.html) API, provides a lot of functional, nn, optimizer interfaces. The API usage and functions are consistent with the mainstream usage in the industry, which is convenient for users to refer to and use. The mint interface is currently an experimental interface and performs better than ops in `jit_level="O0"` and pynative mode. Currently, the graph sinking mode and CPU/GPU backend are not supported, and it will be gradually improved in the future.

  | mindspore.mint  |  |   | |
  |:----|:----|:----|:----|
  | mindspore.mint.eye |mindspore.mint.rand_like|mindspore.mint.isfinite|mindspore.mint.any|
  | mindspore.mint.ones |mindspore.mint.rand|mindspore.mint.log|mindspore.mint.greater_equal|
  | mindspore.mint.ones_like |mindspore.mint.gather|mindspore.mint.logical_and|mindspore.mint.all|
  | mindspore.mint.zeros |mindspore.mint.permute|mindspore.mint.logical_not|mindspore.mint.mean|
  | mindspore.mint.zeros_like |mindspore.mint.repeat_interleave|mindspore.mint.logical_or|mindspore.mint.prod|
  | mindspore.mint.arange |mindspore.mint.abs|mindspore.mint.mul|mindspore.mint.sum|
  | mindspore.mint.broadcast_to |mindspore.mint.add|mindspore.mint.neg|mindspore.mint.eq|
  | mindspore.mint.cat |mindspore.mint.clamp|mindspore.mint.negative|mindspore.mint.ne|
  | mindspore.mint.index_select |mindspore.mint.cumsum|mindspore.mint.pow|mindspore.mint.greater|
  | mindspore.mint.max |mindspore.mint.atan2|mindspore.mint.reciprocal|mindspore.mint.gt|
  | mindspore.mint.min |mindspore.mint.arctan2|mindspore.mint.rsqrt|mindspore.mint.isclose|
  | mindspore.mint.scatter_add |mindspore.mint.ceil|mindspore.mint.sigmoid|mindspore.mint.le|
  | mindspore.mint.narrow |mindspore.mint.unique|mindspore.mint.sin|mindspore.mint.less_equal|
  | mindspore.mint.nonzero |mindspore.mint.div|mindspore.mint.sqrt|mindspore.mint.lt|
  | mindspore.mint.normal |mindspore.mint.divide|mindspore.mint.square|mindspore.mint.maximum|
  | mindspore.mint.tile |mindspore.mint.erf|mindspore.mint.sub|mindspore.mint.minimum|
  | mindspore.mint.topk |mindspore.mint.erfinv|mindspore.mint.tanh|mindspore.mint.inverse|
  | mindspore.mint.sort |mindspore.mint.exp|mindspore.mint.bmm|mindspore.mint.searchsorted|
  | mindspore.mint.stack |mindspore.mint.floor|mindspore.mint.matmul|mindspore.mint.argmax|
  | mindspore.mint.where |mindspore.mint.flip|mindspore.mint.split|mindspore.mint.cos|
  | mindspore.mint.less |||

  | mindspore.mint.nn|
  |:----|
  | mindspore.mint.nn.Dropout  |
  | mindspore.mint.nn.Unfold |
  | mindspore.mint.nn.Fold |
  | mindspore.mint.nn.Linear|
  | mindspore.mint.nn.BCEWithLogitsLoss |

  | mindspore.mint.nn.functional||
  |:----|:----|
  |mindspore.mint.nn.functional.batch_norm |mindspore.mint.nn.functional.group_norm|
  |mindspore.mint.nn.functional.fold |mindspore.mint.nn.functional.layer_norm|
  |mindspore.mint.nn.functional.max_pool2d |mindspore.mint.nn.functional.linear|
  |mindspore.mint.nn.functional.binary_cross_entropy |mindspore.mint.nn.functional.unfold|
  |mindspore.mint.nn.functional.sigmoid |mindspore.mint.nn.functional.one_hot|
  |mindspore.mint.nn.functional.tanh |mindspore.mint.nn.functional.elu|
  |mindspore.mint.nn.functional.binary_cross_entropy_with_logits |mindspore.mint.nn.functional.gelu|
  |mindspore.mint.nn.functional.dropout|mindspore.mint.nn.functional.leaky_relu|
  |mindspore.mint.nn.functional.embedding  |mindspore.mint.nn.functional.silu|
  |mindspore.mint.nn.functional.grid_sample|mindspore.mint.nn.functional.softplus|
  |mindspore.mint.nn.functional.relu|mindspore.mint.nn.functional.softmax|
  |mindspore.mint.nn.functional.pad||

  | mindspore.mint.optim |
  |:----|
  | mindspore.mint.optim.AdamW |

  | mindspore.mint.linalg |
  |:----|
  | mindspore.mint.linalg.inv |

### Non-compatible Interface Changes

- Interface name: `Profiler`

  Changes: The performance data file generated by parsing is streamlined to save space. Delete the FRAMEWORK directory data and other redundant data after exporting the performance data. Retain only the deliverables of the profiler and the original performance data in the PROF_XXX directory to save space. Data simplification mode can be turned off by configuring the `data_simplification` parameter to `False`, which will be consistent with the performance data files generated by the historical version.

- Interface name: The `saved_data` field in the configuration file of the dump function is `"tensor"`.

  Changes: The name of the file to be dumped to disks is changed. `"/"` is replaced with `"_"`, and the operator name is changed to the global name of the operator.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.1 interface </td>
  </tr>
  <tr>
  <td><pre>
  File name format:
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.{format}.npy
  </br>
  Example:
  Conv2D.Conv2D-op12.0.0.1623124369613540.
  output.0.DefaultFormat.npy
  </pre>
  </td>
  <td><pre>
  File name format:
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.{format}.npy
  </br>
  Example:
  Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3
  -Conv2d_Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy
  </pre>
  </td>
  </tr>
  </table>

- Interface name: The `saved_data` field in the Dump function configuration file is `"statistic"`.

  Changes: By default, `'max'`, `'min'`, `'avg'`, `'count'`, `'negative zero count'`, `'positive zero count'`, `'nan count'`,  `'negative inf count'` ,`'positive inf count'`,`'zero count'` and `'md5'`. In the 2.3 version, the `'max'`, `'min'`, and `'l2norm'` statistical items are saved by default. You can customize statistical items by configuring `'statistic_category'`.

### Contributors

caifubi;candanzg;ccsszz;chaiyouheng;changzherui;chenfei_mindspore;chengbin;chengfeng27;Chong;dairenjie;DavidFFFan;DeshiChen;dingjinshan;douzhixing;emmmmtang;Erpim;fary86;fengyixing;fuhouyu;gaoyong10;GuoZhibin;guozhijian;halo;haozhang;hejianheng;Henry Shi;horcham;huandong1;huangbingjian;Jackson_Wong;jiangchenglin3;jiangshanfeng;jiangzhenguang;jiaorui;bantao;jiaxueyu;jijiarong;JuiceZ;jxl;kairui_kou;lanzhineng;LiangZhibo;lichen;limingqi107;linqingke;liubuyu;liujunzhu;liuluobin;liyan2022;liyejun;LLLRT;looop5;lujiale;luochao60;luoyang;lvxudong;machenggui;maning202007;Margaret_wangrui;master_2;mengyuanli;moran;Mrtutu;NaCN;nomindcarry;panzhihui;pengqi;qiuyufeng;qiuzhongya;Renyuan Zhang;shaoshengqi;Shawny;shen_haochen;shenhaojing;shenwei41;shij1anhan;shilishan;shiziyang;shunyuanhan;shuqian0;TAJh;tanghuikang;tan-wei-cheng;Thibaut;tianxiaodong;TronZhang;TuDouNi;VectorSL;wang_ziqi;wanghenchang;wangjie;weiyang;wudawei;wujiangming;wujueying;XianglongZeng;xiaotianci;xiaoxin_zhang;xiaoxiongzhu;xiaoyao;XinDu;xuxinglei;yangchen;yanghaoran;yanglong;yangruoqi713;yangzhenzhang;yangzishuo;Yanzhi_YI;yao_yf;yefeng;yide12;YijieChen;YingLai Lin;yuchaojie;YuJianfeng;zangqx;zhaiyukun;zhangminli;zhangqinghua;ZhangZGC;zhengxinQian;zhengzuohe;zhouyaqiang0;zhuguodong;zhupuxu;zichun_ye;zjun;zlq2020;ZPaC;zuochuanyong;zyli2020;阿琛;狄新凯;范吉斌;冯一航;胡彬;宦晓玲;黄勇;康伟;雷仪婧;李良灿;李林杰;刘崇鸣;刘力力;刘勇琪;刘子涵;吕浩宇;王禹程;熊攀;徐安越;徐永飞;俞涵;张王泽;张栩浩;郑裔;周莉莉;周先琪;朱家兴;邹文祥

Contributions of any kind are welcome!

## MindSpore 2.3.0-rc2 Release Notes

### Major Features and Improvements

#### AutoParallel

- [STABLE] Transpose/Sub/Add/Mul/Div/ReLU/Softmax/Sigmoid supports layout configuration.
- [STABLE] The collective communication precision will affect network convergence. The configuration item [force_fp32_communication](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/mindspore/mindspore.set_auto_parallel_context.html) is provided in the interface mindspore.set_auto_parallel_context. When set to True, the communication type of the reduce communication operator can be forced to be converted to float32.
- [BETA] Pipeline parallel support Interleave. Optimize the performance when micro batch is limited.
- [BETA] Optimize checkpoint transformation speed when using pipeline parallel, support single stage transform.

#### PyNative

- [BETA] Support [recompute](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/mindspore/mindspore.recompute.html) on PyNative mode.
- [STABLE] Support [register_hook](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook) on PyNative mode.

### API Change

Add timeout environment variables in [dynamic networking](https://www.mindspore.cn/tutorials/experts/en/r2.3.0rc2/parallel/dynamic_cluster.html) scenarios:

- `MS_TOPO_TIMEOUT`: Cluster networking phase timeout time in seconds.
- `MS_NODE_TIMEOUT`: Node heartbeat timeout in seconds.
- `MS_RECEIVE_MSG_TIMEOUT`: Node timeout for receiving messages in seconds.

Added new environment variable `MS_ENABLE_LCCL` to support the use of LCCL communication library.

### Bug Fixes

- [#I9CR96](https://gitee.com/mindspore/mindspore/issues/I9CR96) Fix the issue of insufficient timeout time causing failure for dynamic networking startup in large-scale clusters.
- [#I94AQQ](https://gitee.com/mindspore/mindspore/issues/I94AQQ) Fixed the problem of incorrect output shape of ops.Addcdiv operator in graph mode.

### Contributors

Thanks goes to these wonderful people:

bantao,caifubi,changzherui,chenfei_mindspore,chenweifeng,dairenjie,dingjinshan,fangzehua,fanyi20,fary86,GuoZhibin,hanhuifeng,haozhang,hedongdong,Henry Shi,huandong1,huangbingjian,huoxinyou,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jxl,kairui_kou,lichen,limingqi107,liuluobin,LLLRT,looop5,luochao60,luojianing,maning202007,NaCN,niyuxin94520,nomindcarry,shiziyang,tanghuikang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wudawei,XianglongZeng,xiaoxiongzhu,xiaoyao,yanghaoran,Yanzhi_YI,yao_yf,yide12,YijieChen,YingLai Lin,yuchaojie,YuJianfeng,zangqx,zhanghanLeo,ZhangZGC,zhengzuohe,zhouyaqiang0,zichun_ye,zjun,ZPaC,zyli2020,冯一航,李林杰,刘力力,王禹程,俞涵,张栩浩,朱家兴,邹文祥

Contributions of any kind are welcome!

## MindSpore Lite 2.3.0-rc2 Release Notes

### Major Features and Improvements

- [STABLE] Support the configuration of FlashAttention related properties in the configuration file used by the cloud-side conversion tool.
- [STABLE] Support multi-devices memory sharing.

### Contributors

Thanks goes to these wonderful people:

emmmmtang,熊攀

Contributions of any kind are welcome!

## MindSpore 2.3.0-rc1 Release Notes

### Major Features and Improvements

#### DataSet

- [STABLE] Support integrity check, encryption and decryption check for MindRecord to protect the integrity and security of user data.
- [STABLE] MindRecord api changes: FileWriter.open_and_set_header is deprecated since it has been integrated into FilterWriter, if the old version code reports an error, delete this call; Add type checking for data in FileWriter to ensure that the data type defined by the Schema matches the real data type; The return value of all methods under Mindrecord are removed, replaced by an exception when processing error is occurred.
- [STABLE] Support Ascend processing backend for the following transforms: ResizedCrop, HorizontalFlip, VerticalFlip, Perspective, Crop, Pad, GaussianBlur, Affine.
- [STABLE] Optimized the content of data processing part in model migration guide, providing more examples to compare with third-party frameworks.
- [STABLE] Optimized the parsing efficiency of TFRecordDataset in multiple data columns scenario, improving the parsing performance by 20%.

#### PIJIT

- [BETA]PIJit analyzes and adjusts the Python bytecode and performs graph capture and graph optimization on the execution flow. Supported Python codes are executed in static graph mode, and unsupported ones are divided into subgraphs and executed in dynamic graph mode, automatically achieving dynamic and static unification. Users can enable the PIJit function by decorating the function with @jit(mode="PIJit", jit_config={options:value}).

#### Inference

- [DEMO] The integrated architecture of large model inference, upgrade, training, and promotion unifies scripts, distributed policies, and runtime. The period from training to inference deployment of typical large models is reduced to days. Large operators are integrated to reduce the inference latency and effectively improve the network throughput.

#### AutoParallel

- [STABLE] Add msrun startup method to launch distributed job with single instruction.
- [STABLE] Add to be deprecated hint for RankTable startup method.
- [STABLE] Eliminate redundant constants in graph mode to improve compilation performance and memory overhead.
- [STABLE] The subgraph scenario optimizer parallelizes the first subgraph inline, allowing some computation and communication masking under pipeline parallelism to be performed.
- [STABLE] Communication information export: export model communication information (communication domain, communication volume) during compilation, and input it to the cluster as the basis for communication scheduling.
- [STABLE] Pipeline parallel inference is optimized, eliminates shared weights forwarding between stages, improving execution performance. Supports automatic broadcast of pipeline inference results, improving the usability of autoregressive inference.
- [STABLE] Operator-level parallel sharding supports the configuration of the mapping between the device layout and tensor layout during MatMul/Add/LayerNorm/GeLU/BiasAdd operator sharding.
- [STABLE] Supports gradient communication and backward calculation overlapping in the data parallel dimension.
- [STABLE] Single device simulation compilation, used to simulate the compilation process of a certain device in multi device distributed training, assisting in analyzing the compilation processes and memory usage on the front and back ends.
- [STABLE] Implement ops.Tril sharding to reduce the memory and performance requirements on a single device.
- [BETA] Supports the fusion between communication operators and computing operators, in order to overlap communication overheads with computation and improve network performance.
- [BETA] Load checkpoints and compile graphs in parallel to accelerate fault recovery.

#### Runtime

- [BETA] Support O0/O1/O2 multi-level compilation to improve static graph debugging and tuning capabilities.

#### FrontEnd

- [STABLE] The framework supports the bfloat16 data type. dtype=mindspore.bfloat16 can be specified when a tensor is created.
- [STABLE] The syntax support capability of the rewrite component is optimized, syntaxs such as class variables, functions, and control flows can be parsed.
- [STABLE] New context setting: debug_level. User can use mindspore.set_context(debug_level=mindspore.DEBUG) to get more debug information.

#### Profiler

- [BETA] Dynamically start and stop profiling. Users can collect profiling data in real time according to the training situation, reducing the amount of data collected.
- [BETA] Profiling the communication operator time-consuming matrix. Users can find cluster communication performance bottlenecks by analyzing the communication operator time-consuming matrix.
- [BETA] Improve the performance of Ascend environment in parsing profiling data.
- [BETA] Supports offline analysis of data generated by Profiling. Users can collect data first and then parse the data as needed.
- [BETA] Supports collecting performance data of On-Chip Memory, PCIe, and l2_cache to enrich performance analysis indicators.

#### Dump

- [BETA] The statistical information saved by Dump records MD5 values, and users can determine small differences in tensor values through MD5 values.
- [BETA] Dump supports the float16 data type and supports users to locate float16 type operator accuracy issues.

#### PyNative

- [STABLE] Reconstruct the single operator calling process for dynamic graphs to improve the performance of dynamic graphs.

#### Ascend

- [BETA] Support set configuration options of CANN, which are divided into two categories: global and session. Users can configure them through mindspore.set_context(Ascend_configuration={"ge_options": {"global": {"global_option": "option_value"}, "session": {"session option": "option_value"}}).

#### API Change

- Add mindspore.hal API to support stream, event, and device management capabilities.
- Add mindspore.multiprocessing API to provide the capability of creating multiple processes.

#### Operators

- [BETA] mindspore.ops.TopK now supports the second input k as an int32 type tensor.

### Bug Fixes

- [#I92H93] Fixed the issue of 'Launch kernel failed' when using the Print operator to print string objects on the Ascend platform.
- [#I8S6LY] Fixed RuntimeError: Attribute dyn_input_sizes of Default/AddN-op1 is [const vector]{}, of which size is less than 0 error of variable-length input operator, such as AddN or Concat, for dynamic shape process in graph mode on the Ascend platform.
- [#I9ADZS] Fixed the data timeout issue in network training due to inefficient dataset recovery in the fault recovery scenario.

### Contributors

Thanks goes to these wonderful people:

AlanCheng511，AlanCheng712，bantao，Bingliang，BJ-WANG，Bokai Li，Brian-K，caifubi，cao1zhg，CaoWenbin，ccsszz，chaiyouheng，changzherui，chenfei_mindspore，chengbin，chengfeng27，chengxb7532，chenjianping，chenkang，chenweifeng，Chong，chuht，chujinjin，Cynthia叶，dairenjie，DavidFFFan，DeshiChen，douzhixing，emmmmtang，Erpim，fangzhou0329，fary86，fengxun，fengyixing，fuhouyu，gaoshuanglong，gaoyong10，GaoZhenlong，gengdongjie，gent1e，Greatpan，GTT，guoqi，guoxiaokang1，GuoZhibin，guozhijian，hangq，hanhuifeng，haozhang，hedongdong，hejianheng，Henry Shi，heyingjiao，HighCloud，Hongxing，huandong1，huangbingjian，HuangLe02，huangxinjing，huangziling，hujiahui8，huoxinyou，jiangchenglin3，jianghui58，jiangshanfeng，jiaorui，jiaxueyu，JichenZhao，jijiarong，jjfeing，JoeyLin，JuiceZ，jxl，kairui_kou，kate，KevinYi，kisnwang，lanzhineng，liangchenghui，LiangZhibo，lianliguang，lichen，ligan，lihao，limingqi107，ling，linqingke，liruyu，liubuyu，liuchao，liuchengji，liujunzhu，liuluobin，liutongtong9，liuzhuoran2333，liyan2022，liyejun，LLLRT，looop5，luochao60，luojianing，luoyang，LV，machenggui，maning202007，Margaret_wangrui，MaZhiming，mengyuanli，MooYeh，moran，Mrtutu，NaCN，nomindcarry，panshaowu，panzhihui，PingqiLi，qinzheng，qiuzhongya，Rice，shaojunsong，Shawny，shenwei41，shenyaxin，shunyuanhan，silver，Songyuanwei，tangdezhi_123，tanghuikang，tan-wei-cheng，TingWang，TronZhang，TuDouNi，VectorSL，WANG Cong，wang_ziqi，wanghenchang，wangpingan，wangshaocong，wangtongyu6，weiyang，WinXPQAQ，wtcheng，wudawei，wujiangming，wujueying，wuweikang，wwwbby，XianglongZeng，xiaosh，xiaotianci，xiaoxin_zhang，xiaoxiongzhu，xiaoyao，XinDu，xingzhongfan，yanghaoran，yangluhang，yangruoqi713，yangzhenzhang，yangzishuo，yanjiaming，Yanzhi_YI，yao_yf，yefeng，yeyunpeng2020，yide12，YijieChen，YingLai Lin，YingtongHu，youshu，yuchaojie，YuJianfeng，zangqx，zby，zhaiyukun，zhangdanyang，zhanghaibo，zhanghanLeo，zhangminli，zhangqinghua，zhangyanhui，zhangyifan，zhangyinxia，zhangyongxian，ZhangZGC，zhanzhan，zhaoting，zhengyafei，zhengzuohe，ZhihaoLi，zhouyaqiang0，zhuguodong，zhumingming，zhupuxu，zichun_ye，zjun，zlq2020，ZPaC，zuochuanyong，zyli2020，陈宇，代宇鑫，狄新凯，范吉斌，冯一航，胡彬，宦晓玲，黄勇，康伟，李良灿，李林杰，刘崇鸣，刘力力，刘勇琪，吕浩宇，没有窗户的小巷，王禹程，吴蕴溥，熊攀，徐安越，徐永飞，许哲纶，俞涵，张峻源，张树仁，张王泽，张栩浩，郑裔，周莉莉，周先琪，朱家兴，邹文祥

Contributions of any kind are welcome!

## MindSpore 2.2.13 Release Notes

### API Change

Add timeout environment variables in dynamic networking scenarios:

- `MS_TOPO_TIMEOUT`: Cluster networking phase timeout time in seconds.
- `MS_CLUSTER_RETRY_NUM`: Number of node's retrying registration during cluster networking phase.
- `MS_NODE_TIMEOUT`: Node heartbeat timeout in seconds.
- `MS_RECEIVE_MSG_TIMEOUT`: Node timeout for receiving messages in seconds.

### Bug Fixes

- [#I9CR96] Fix the issue of insufficient timeout time causing failure for dynamic networking startup in large-scale clusters.

### Contributors

Thanks goes to these wonderful people:

ZPaC, limingqi107, lizhenyu, jiangshanfeng

Contributions of any kind are welcome!

## MindSpore 2.2.12 Release Notes

### Major Features and Improvements

- [STABLE] Optimize scnarios where network parameters are initialized by fp32, and optimizer parallel mode is on, reducing the amount of Cast operator.
- [STABLE] Add detection and processing capabilities to silent fault detection. Silent faults may lead to error during training procedures, this helps users to prevent or lower the cost of fault location, which caused by silent faults.

### Bug Fixes

- [#I97D1L] Fix ReduceLROnPlateau, LRScheduler, CosineAnnealingWarmRestarts dynamic learning rate related interface sample error.
- [#I970HV] Fix the problem where order of AllGather/ReduceScatter between two cards is not preserved.
- [#I99JPI] Fix load checkpoint for bfloat16 parameter during vague load mode.

### Contributors

Thanks goes to these wonderful people:

yao_yf, YijieChen, 冯一航, yuchaojie, 李良灿, YuJianfeng, huangxinjing, GuoZhibin, looop5

Contributions of any kind are welcome!

## MindSpore 2.2.11 Release Notes

### Major Features and Improvements

#### scipy

- [STABLE] Add new API mindspore.scipy.optimize.linear_sum_assignment in scipy module to solve the linear sum assignment problem. It can find the least-cost assignment based on a given cost matrix.

### Bug Fixes

- [#I8JVRU] Fixed the problem where the results of the bernoulli random operator running twice on the GPU are probabilistically consistent.
- [#I8OC32] Fixed the segmentation fault error because the MatrixSetDiagV3 operator does not verify abnormal input.

### Contributors

Thanks goes to these wonderful people:

fary86, wanghenchang, haozhang, mengyuanli, emmmmtang, luoyang, zhupuxu, zhangyongxian, liuluobin, LLLRT, TuDouNi, hujiahui8, wangtongyu6, ligan, zhuguodong, yanghaoran, YingtongHu, liyejun, zjun, 徐永飞, chuht, 张树仁, 徐安越, DeshiChen, shenyaxin, liujunzhu, shunyuanhan, yuchaojie, yao_yf, 没有窗户的小巷, yeyunpeng2020, weiyang, KevinYi, hedongdong, zhouyaqiang0, Margaret_wangrui, zhanghaibo, moran, huangziling, 朱家兴, GuoZhibin, 李良灿, jiaxueyu, gaoyong10, Greatpan, 宦晓玲, melody, 俞涵, jiangshanfeng, XinDu, ling, caifubi, zhangyinxia, gengdongjie, Erpim, XianglongZeng, zhangminli, fengyixing, 冯一航, 黄勇, panzhihui, 胡彬, linqingke, wangshaocong

Contributions of any kind are welcome!

## MindSpore Lite 2.2.11 Release Notes

### Bug Fixes

- [#I8TPLY] Fixed SSD MobileNetV2 FPN network inference error on Atlas inference series products.

### Contributors

Thanks goes to these wonderful people:

wangtongyu6, zhuguodong, 徐永飞, 徐安越, yeyunpeng2020, moran, XinDu, gengdongjie.

Contributions of any kind are welcome!

## MindSpore 2.2.10 Release Notes

### Major Features and Improvements

#### Operators

- [STABLE] FastGelu, BatchMatMul, AllReduce, AllGather, Broadcast, ReduceScatter support bfloat16 data type
- [STABLE] AllGather support uint8 data type

### Bug Fixes

- [#I8ALW3] Fixed networks including Faster R-CNN, DeepText, MaskRCNN-ResNet50, which had errors while training RandomChoiceWithMask operator in Ascend 8P scenario.
- [#I8LKG7] Fixed graph compilation error of UNet-2D in Ascend 1P/8P scenario.
- [#I8KU3X] Fixed CRNN-ResNet34 network, which stuck in training phase in Ascend 1P/8P PyNative mode.
- [#I8KTHH] Fixed BERT network error when training without allreduce grouped fusion with enable_parallel_optimizer=True, in Ascend 8P scenario.

### Contributors

Thanks goes to these wonderful people:

李林杰, TuDouNi, chengxb7532, Henry Shi, rms-infer-type, 朱家兴, zhouyaqiang0, tanghuikang, gaoyong10, gengdongjie, yao_yf, hujiahui8, hanhuifeng, shenyaxin, KevinYi, 冯一航, chengfeng27, JuiceZ, zhangyanhui, jijiarong, xiaoxiongzhu, 没有窗户的小巷, ling, liyan2022, haozhang, zangqx, xiaoyao, liujunzhu, 胡彬, panzhihui, wangshaocong, linqingke, jianghui58, qiuzhongya, yangruoqi713, zhangminli, moran, 王禹程, shaojunsong, wangtongyu6, zhupuxu, luoyang, 徐安越, qinzheng, caifubi, 徐永飞, chenkang, youshu, XinDu, liubuyu, jxl, yeyunpeng2020, huoxinyou, yefeng, jiaorui, wangpingan, cao1zhg, zjun, zyli2020, yanjiaming, Cynthia叶, 胡安东, 李良灿, liruyu, liuluobin, lihao, huangbingjian, YijieChen, jjfeing, looop5, 刘力力, xiaoxin_zhang, yangluhang, chenweifeng, jiangshanfeng, zichun_ye, 陈宇, NaCN, ligan, YingLai Lin, huangziling, chenjianping, DeshiChen, chengbin, kairui_kou, ccsszz, yanghaoran, zhangdanyang, Yanzhi_YI, zhengzuohe, hangq, TronZhang, wanghenchang, HighCloud, 吕浩宇, VectorSL, ZPaC, mengyuanli, maning202007, 刘勇琪, r1chardf1d0, fary86, 刘崇鸣, yuchaojie, douzhixing, fengyixing

Contributions of any kind are welcome!

## MindSpore Lite 2.2.10 Release Notes

### Bug Fixes

- [#I8K7CC] Optimize error message when non-string segments are passed to get_model_info.

### Contributors

Thanks goes to these wonderful people:

gengdongjie, zhangyanhui, xiaoxiongzhu, wangshaocong, jianghui58, moran, wangtongyu6, 徐安越, qinzheng, 徐永飞, youshu, XinDu, yeyunpeng2020, yefeng, wangpingan, zjun, 胡安东, 刘力力, 陈宇, chenjianping, kairui_kou, zhangdanyang, hangq, mengyuanli, 刘崇鸣

Contributions of any kind are welcome!

## MindSpore 2.2.1 Release Notes

### Bug Fixes

- [#I7R3R5] Fixed the problem that the network precision of the ResNet-50 on the Ascend platform deteriorates.
- [#I8A9RH] Fixed an issue where the DBNet(ResNet-50) network precision on the Ascend platform deteriorates.
- [#I8B8IW] Fixed the segment error caused by out-of-bounds multi-dimensional tensor assignment.
- [#I8J0F4] Fixed an issue where the multidimensional Tensor extension dimension fails to be executed in the dynamic graph.
- [#I87P3P] Fixed an issue where the compilation cache fails to be loaded during secondary training on the Ascend platform.
- [#I86GP9] Fixed an issue where the UNet3D network inference precision deteriorates on the Ascend platform.
- [#I89B4K] Fixed an issue where the dynamic rank execution of dynamic graphs on the Windows platform is suspended.
- [#I8CX0C] Fixed an issue where dynamic images occasionally fail in mixed precision mode on the Ascend platform.
- [#I8BGCF] Fixed an issue where a segment error occurs when the command is executed in dynamic diagram mode of the AirNet network on the Ascend platform.
- [#I8L5DS] Fixed an issue where the ResNet-50 image segmentation network dynamic image is executed slowly on the Ascend platform.

### Contributors

Thanks goes to these wonderful people:

yufan, dingcheng, lvzhangcheng, zhunaipan, fangwenyi, weiyang, changzherui, chujinjin, zangqingxiang, yuchaojie, wuweikang, tanghuikang, xiaoyao, huangbinjian, zhoupeichen, chenfei_mindspore, hedongdong, wangnan, zhengzuohe, yanghaoran, zouliqin, luoyang, liuchongmin, lujiale, machenggui, wangcong, lixiangyi, wangting, huangyong

Contributions of any kind are welcome!

## MindSpore Lite 2.2.1 Release Notes

### Bug Fixes

- [#I88055] Fixed a function issue caused by incorrect format setting of the gridsample operator in MindSpore Lite inference.
- [#I8D80Y] The MindSpore Lite inference single-operator invoking process resources are not released and exits abnormally.

### Contributors

Thanks goes to these wonderful people:

zhanghaibo, wangsiyuan, wangshaocong, chenjianping

Contributions of any kind are welcome!

## MindSpore 2.2.0 Release Notes

### Major Features and Improvements

#### DataSet

- [STABLE] The `row_size` parameter of data operation map/batch is extended to support passing list, which stands for [Input Shared Memory, Output Shared Memory], so as to flexibly control the size of shared memory in multi-process mode.
- [STABLE] Provide 100% mindspore.dataset and mindspore.dataset.transforms samples for reference.
- [STABLE] ConcatDataset supports global sampling. After combining data from multiple sources using concat operation, data can be globally sampled randomly to enhance data diversity.
- [STABLE] When the model.train API is used for training, TimeMonitor(.., data_time=True) can be used to monitor data processing performance in real time.
- [STABLE] Introduced the jemalloc library to solve the problem of slow memory rise due to untimely memory debris recovery in extreme scenarios.

#### FrontEnd

- [STABLE] Support adding decorator @lazy_inline to make a graph generated from cell being inlined lazily, which can improve the compilation performance effectively.
- [STABLE] Optimize the function of mixed precision training, support automatic rewriting of Python scripts through rewrite to achieve mixed precision strategies, and support automatic parsing of functions, branch statements, and other syntax.
- [STABLE] Mixed precision function optimization, ReWrite supports syntax parsing of class functions and branch statements, and extends O1 functionality.
- [STABLE] Optimize the dynamic learning rate function and add APIs such as MultiStepLR; function get_lr and global_step decoupling, extending optimizer module functionality.
- [STABLE] Optimize API code samples, API difference tables, and tutorials for using higher-order functions.

#### Operator

- [STABLE] Add new operator primitive `mindspore.ops.Dense`.
- [STABLE] Add the random number operator state management feature, which allows the random number operator to save the state of the random number, and can be stably reproduced in scenarios such as model parallelism and recalculation. Currently, it only supports CPU/GPU platforms, and the involved random number operators include: `mindspore.ops.Multinomial`, `mindspore.ops.MultinomialWithReplacement`, `mindspore.ops.ParameterizedTruncatedNormal`, `mindspore.ops.StandardLaplace`, `mindspore.ops.StandardLaplace`, `mindspore.ops.Uniform`, `mindspore.ops.UniformInt`, `mindspore.ops.UniformReal`, `mindspore.ops.UniformInt`, `mindspore.ops.Dropout`, `mindspore.ops.RandomChoiceWithMask`, `mindspore.ops.RandomCategorical`, `mindspore.ops.RandomShuffle`, `mindspore.ops.RandamGamma`, `mindspore.ops.RandomPoisson` and `mindspore.ops.TruncatedNormal`.
- [STABLE] When a GPU operator encounters an illegal input scenario, it supports asynchronously printing error logs in the CUDA kernel of the operator to the Host side and interrupting the execution of the current CUDA Stream, improving the efficiency of user operator problem positioning.

#### PyNative

- [STABLE] Support viewing mechanism in PyNative mode.
- [STABLE] Function enhancement in PyNative mode: sens supports dict input type.

#### Ascend

- [STABLE] Supports user configurable operator high-precision/high-performance mode, users can use `context.set_context(ascend_config={"op_precision_mode": "/path/to/op_precision_config_file"})` to configure high-precision/high-performance modes for some TBE operators.
- [BETA] Supports user configurable operators for fp16-in and fp32-out, users can use `context.set_context(ascend_config={"precision_mode": "force_fp32"})` to configure fp16-in and fp32-out for the TBE Cube operators.
- [BETA] Remove the strong binding between `jit_level="O3"` and GE processes, so users no longer need to set `jit_level="O3"` when executing GE processes.

#### Parallel

- [STABLE] Support the gradient accumulation feature in non-pipeline parallel scenarios in semi-automatic/fully automatic mode. Users can enable gradient accumulation by writing `net = GradAccumulationCell(net, micro_size)`. The gradient accumulation feature is compatible with the  lazy_inline feature.

#### Inference

Since version 2.2, the MindSpore main release package does not provide the inference interface enabling for the Ascend 310. If you need to use the inference interface, install the MindSpore Lite release package or download the MindSpore version earlier than 2.0. For details about how to install and use MindSpore Lite, see <https://www.mindspore.cn/lite/en>. HUAWEI Ascend 310 (Ascend) is an energy-efficient and highly integrated AI processor for edge scenarios. It supports inference on MindIR models. In the earlier version, MindSpore provides two methods for enabling inference on the Ascend 310 hardware:

1. The MindSpore main release package provides the matching Ascend 310 version that supports C++ inference interfaces.
2. The MindSpore Lite release package provides the matching Ascend version and supports C++ and Java inference.

The C++ APIs provided by the two solutions are basically the same. In the future, MindSpore Lite is used instead of building and maintaining two sets of interfaces. The original 310 inference service built based on the MindSpore main release package can be switched to MindSpore Lite with a few modifications. For details, see <https://www.mindspore.cn/docs/en/master/faq/inference.html>.

### Bug fixes

- [I7SDA0] Fixed an issue where the accuracy of the CRNN network deteriorates on the NES platform.
- [I7T4QK] Fixed an issue where the inference precision of the WGAN network deteriorates on the OptiX OSN 8800 platform.
- [I7TJ8Z] Fixed an issue where the inference precision of the LGTM network deteriorates on the OptiX OSN 8800 platform.
- [I7M58O] Fixed ASR-dynamic network training core dump issue on Ascend platform.
- [I7L6B6] Fixed an issue where child processes do not exit in some scenarios when dataset is in multi-process mode.
- [I7L7AE] Fixed an issue where dataset pipeline contains repeat operations and dynamic batchinfo.get_epoch_num() is incorrectly used in dataset.batch.
- [I7UY7G] Rectify the file permission modification error in OBSMindDataset.

### Contributors

Thanks goes to these wonderful people:
bantao, Bingliang, BJ-WANG, Brian-K, caifubi, ccsszz, changzherui, chenfei_mindspore, chengfeng27, chenhaozhe, chenjianping, chenkang, chenweifeng, chuht, chujinjin, CShu0507, Cynthia叶, DeshiChen, douzhixing, Erpim, Etienne, fary86, fengxun, fengyixing, gaoshuanglong, Gaoxiong, gaoyong10, GaoZhenlong, Greatpan, GuoZhibin, guozhijian, hangq, hanhuifeng, haozhang, hedongdong, Henry Shi, HighCloud, Hongxing, huangbingjian, huanghui, huangxinjing, huangziling, hujiahui8, huoxinyou, HWalkingMan, jianghui58, jiangshanfeng, jiaorui, jijiarong, jjfeing, JuiceZ, jxl, KevinYi, kisnwang, KXiong, lanzhineng, Li Qingguo, LiangZhibo, lianliguang, ligan, lihao, Lihoon, limingqi107, ling, linqingke, liruyu, liubuyu, liuchao, liujunzhu, liuluobin, liupeng303, liutongtong9, liyan2022, liyejun, looop5, luochao60, luojianing, luoyang, machenggui, maning202007, Margaret_wangrui, MaZhiming, mengyuanli, moran, NaCN, nomindcarry, panshaowu, panzhihui, qinzheng, qiuzhongya, r1chardf1d0, shaojunsong, shenwei41, shenyaxin, shenzhangyi, Shira Zaloshinski, shunyuanhan, tangdezhi_123, tanghuikang, tan-wei-cheng, tan-wei-cheng-3260, TronZhang, TuDouNi, VectorSL, wang_ziqi, wanghenchang, wangpingan, wangshaocong, wangtongyu6, wtcheng, wujueying, XianglongZeng, xiaotianci, xiaoxin_zhang, xiaoxiongzhu, xiaoyao, xiaoyuanyuan, XinDu, xujinliang, xupan, yanghaoran, yangluhang, yangruoqi713, yangsijia, yangzhenzhang, yangzishuo, yanjiaming, Yanzhi_YI, yao_yf, yefeng, yeyunpeng2020, yide12, YijieChen, YingLai Lin, YingtongHu, yonibaehr, youshu, yuchaojie, YuJianfeng, zangqx, zhaizhiqiang, zhangbuxue, zhangchunlei, zhangdanyang, zhangdong, zhanghaibo, zhangminli, zhangqi, zhangqinghua, zhangyanhui, zhangyifan, zhangyongxian, zhangzhen, zhangzheng, zhanzhan, zhengzuohe, ZhihaoLi, zhoufeng, zhouyaqiang0, zhuguodong, zhupuxu, zichun_ye, zjun, ZPaC, zuochuanyong, zyli2020, 陈宇, 程超, 范吉斌, 冯浩, 冯一航, 胡彬, 宦晓玲, 黄勇, 雷元哲, 黎冠新, 李良灿, 李林杰, 刘崇鸣, 刘力力, 刘思铭, 刘勇琪, 吕浩宇, 没有窗户的小巷, 沈竞兴, 王禹程, 王振邦, 徐安越, 徐永飞, 俞涵, 张澍坤, 周超, 朱家兴

Contributions of any kind are welcome!

## MindSpore Lite 2.2.0 Release Notes

### Major Features and Improvements

#### FlashAttention Operator Fusion

- [STABLE] The OptiX OSN Ascend series supports the FlashAttention large operator fusion of the LLAMA and stable diffusion models.

## MindSpore 2.1.1 Release Notes

### Bug fixes

- [I7Q9RX] The Ascend platform supports adaptive identification of different hardware types.
- [I7SDA0] Fixed an issue where the accuracy of the CRNN network deteriorates on the NES platform.
- [I7T4QK] Fixed an issue where the inference precision of the WGAN network deteriorates on the OptiX OSN 8800 platform.
- [I7TJ8Z] Fixed an issue where the inference precision of the LGTM network deteriorates on the OptiX OSN 8800 platform.

### Contributors

Thanks goes to these wonderful people:

changzherui, chenfei_mindspore, chenjianping, chenkang, chenweifeng, chujinjin, fangwenyi, GuoZhibin, guozhijian, hangq, hanhuifeng, haozhang, hedongdong, You Shu, Zhou Feng, Dai Yuxin

Contributions of any kind are welcome!

## MindSpore Lite 2.1.1 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite Cloud Inference adds support for Python 3.8 and Python 3.9

## MindSpore 2.1.0 Release Notes

### Major Features and Improvements

#### FrontEnd

- [BETA] JIT Fallback supports variable scenarios. In static graph mode, JIT Fallback supports return of Dict type and Scalar type, supports property setting of non-Parameter type objects, supports partial in-place modification operations of List, and supports third-party libraries such as NumPy. Moreover, it supports related operations of user-defined classes and supports Python basic operators and built-in functions to use more data types. It is compatible with features like control flow, side effects, automatic differentiation. For more details, please refer to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax_support.html).

- [BETA] In static graph mode, the error message of using undefined variables in the control flow scene is optimized. When using variables defined in if, while, and for control flow branches, the variables need to be initialized and defined before the control flow.

- [STABLE] Add module ReWrite, support the ability to modify multiple network in batches based on customized rules.

- [BETA] Add optim_ex module for optimizers, extend the current functionality, support parameter grouping for every parameter in the optimizer, and support parameter modification by assignment while training.

- [STABLE] Optimize PyTorch and MindSpore API Mapping Table, specify the differences between APIs among functionality, parameter, input, output and specialized cases.

#### PyNative

- Optimize the performance of dynamic shape scenes in PyNative mode.

#### DataSet

- [STABLE] Optimize the memory structure of MindRecord data files. Memory consumption can be reduced 60% when loading 100TB+ data for training.
- [STABLE] Support single-thread execution of data processing pipeline, and users can add code in the data pipeline for debugging.
- [STABLE] Optimize the performance of TFRecordDataset to improve the performance of dataset loading by 60%+. Optimize the performance of batch to improve the performance by 30% for the scenarios with large number of batch.
- [STABLE] Optimize API documentation of [mindspore.dataset](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.html) and [mindspore.dataset.transforms](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html). Four new sample libraries have been added to show the effect of data enhancement, namely: [Load & Process Datasets Using Data Pipeline](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.html#quick-start-of-dataset-pipeline), [Visual Transformation Sample Library](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.vision), [Text Transform Sample Library](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.text), [Audio Transform Sample Library](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.audio)

#### AutoParallel

- [STABLE] Support offload parameters or intermediate activations to the CPU or NVMe storage during training process. Users can enable this offload feature by configuring context to scale up the trainable model size.

- [STABLE] Enhanced automatic parallel capability including:

  1. Performance of automatic strategy for typical networks is no less than 90% of default configuration.

  2. Support 3D hybrid parallel training: automatic operator-level strategy generation combined with manual configured pipeline partition.

#### Runtime

- [STABLE] Upgrade OpenMPI version to 4.1.4.
- [STABLE] Upgrade NCCL version to 2.16.5.
- [STABLE] Assign rank id continuously in same node when using dynamic cluster to launch distributed jobs.
- [STABLE] No adaptation code is required for Scheduler node. The script of Scheduler could be identical to that of Worker.

#### Ascend

- [STABLE] Support dump assisted debug information for operator AIC Error scenario. The information includes the operator task name, stream ID, input/output/workspace address and so on.
- [STABLE] Provide default processing mechanism, which skips its execution,  for CANN operators for empty Tensor output scenarios.
- [STABLE] Supplement debug information when network model fails to execute in graph mode. The debug information will saved in a CSV file in rank_${id}/exec_order/, recording the task ID and stream ID of each task.

#### Profiler

- [STABLE] The Profiler supports the collection of time-consuming data from all phases on the Host side.
- [BETA] The Profiler supports the collection of memory data from all phases on the Host side.
- [BETA] The Profiler supports the collection of data processing operator time consumption.

### API Change

- `mindspore.set_context` adds `jit_syntax_level` option, which is used to set JIT syntax support level. For more details, please refer to [set_context](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.set_context.html).
- Change the `model.infer_predict_layout` interface, which has a new parameter skip_backend_compile with a default value of False. Set to True when the user wants to skip the backend compilation process to get the parameter slicing strategy.

#### Operators

- Add operator primitive for `mindspore.ops.ApplyAdamWithAmsgradV2`. It is recommended to call this operator through API `mindspore.nn.Adam`.
- Add operator primitive for `mindspore.ops.UpsampleTrilinear3D`. It is recommended to call this operator through API `mindspore.ops.interpolate`.
- Add operator primitive for `mindspore.ops.UpsampleNearest3D`. It is recommended to call this operator through API `mindspore.ops.interpolate`.

#### API Deprecation

- Deprecate operator primitive `mindspore.ops.ScatterNonAliasingAdd`. It is recommended to use operator primitive `mindspore.ops.TensorScatterAdd` as a replacement.

#### Backwards Incompatible Change

- Interface name: `mindspore.nn.Dense`, `mindspore.nn.Conv1d`, `mindspore.nn.Conv1dTranspose`, `mindspore.nn.Conv2d`, `mindspore.nn.Conv2dTranspose`, `mindspore.nn.Conv3d`, `mindspore.nn.Conv3dTranspose`

  Changes: Change initialization parameter strategy. The default value of weight_init is changed from "normal" to None, and the default value of bias_init is changed from "zeros" to None.

  Description: The default initialization method for weights has been changed from "normal" to internal HeUniform initialization. The default initialization method of bias is changed from "zeros" to internal Uniform initialization.

  <table>
  <tr>
  <td style="text-align:center"> Original interface </td> <td style="text-align:center"> v2.1 interface </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Dense(in_channels,
                     out_channels,
                     weight_init='normal',
                     bias_init='zeros',
                     has_bias=True,
                     activation=None)
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Dense(in_channels,
                     out_channels,
                     weight_init=None,
                     bias_init=None,
                     has_bias=True,
                     activation=None)
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init='normal',
                      bias_init='zeros')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init=None,
                      bias_init=None)
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv1dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               dilation=1,
                               group=1,
                               has_bias=False,
                               weight_init='normal',
                               bias_init='zeros')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv1dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               dilation=1,
                               group=1,
                               has_bias=False,
                               weight_init=None,
                               bias_init=None)
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv2d(in_channels,
                      out_channels, kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init='normal',
                      bias_init='zeros',
                      data_format='NCHW')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init=None,
                      bias_init=None,
                      data_format='NCHW')
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv2dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               output_padding=0,
                               dilation=1,
                               group=1,
                               has_bias=False,
                               weight_init='normal',
                               bias_init='zeros')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv2dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               output_padding=0,
                               dilation=1,
                               group=1,
                               has_bias=False,
                               weight_init=None,
                               bias_init=None)
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init='normal',
                      bias_init='zeros',
                      data_format='NCDHW')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad_mode='same',
                      padding=0,
                      dilation=1,
                      group=1,
                      has_bias=False,
                      weight_init=None,
                      bias_init=None,
                      data_format='NCDHW')
  </pre>
  </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.nn.Conv3dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               dilation=1,
                               group=1,
                               output_padding=0,
                               has_bias=False,
                               weight_init='normal',
                               bias_init='zeros',
                               data_format='NCDHW')
  </pre>
  </td>
  <td><pre>
  mindspore.nn.Conv3dTranspose(in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               pad_mode='same',
                               padding=0,
                               dilation=1,
                               group=1,
                               output_padding=0,
                               has_bias=False,
                               weight_init=None,
                               bias_init=None,
                               data_format='NCDHW')
  </pre>
  </td>
  </tr>
  </table>

### Bug Fixes

- [I6TKLW] Fix the issue of MobileNetV2 network performance degradation on the Ascend platform.
- [I7CP5H] Fix the issue where ASR network training failed on the Ascend platform.
- [I7I3EZ] Fix the issue that caused run_check() failure due to changes to the enumeration interface in Pillow version 10.0.0. If encountered in a lower version of MindSpore, install versions of Pillow below 10.0.0 to avoid this issue.
- [I7IZ8K] Fix accuracy issues with the assignsub interface in PyNative mode.
- [I7HGY0] Fix the issue that the loss of the functional programming does not converge in the PyNative data_sink mode.
- [I7J4N3] Fix the issue that the generation of Step Trace failed in Profiler dynamic Shape mode
- [I7J4N3] Fix the issue that there is no data displayed in the MindInsight parallel strategy view.
- [I79YY4] Fix SiLU operator error when high-order differential in PyNative mode.
- [I6NQJQ] Fix the issue of probabilistic failure in dynamic shape scenarios of the ScatterUpdate operator in PyNative mode.
- [I6Y4G5] Fix the issue of failure in dynamic Shape scenarios of the Conv3D operator in Graph mode.

### Contributors

Thanks goes to these wonderful people:

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

Contributions of any kind are welcome!

## MindSpore Lite 2.1.0 Release Notes

### Major Features and Improvements

#### MindSpore Lite Cloud Inference

- [STABLE] Supports high-performance inference for single-device large model and single-node multi-device distributed large model at Ascend backend.
- [STABLE] Python API Ascend backend supports multiple models sharing workspace memory.
- [STABLE] [The weights can be shared by multiple models through ModelGroup](https://mindspore.cn/lite/docs/en/r2.1/use/cloud_infer/runtime_cpp.html#multiple-models-sharing-weights). For example, weights can be shared between full models and incremental models in the large model scenario.

#### API

The [Python](https://www.mindspore.cn/lite/api/en/r2.1/mindspore_lite/mindspore_lite.ModelGroup.html) and [C++](https://mindspore.cn/lite/api/en/r2.1/generate/classmindspore_ModelGroup.html) ModelGroup interface is added. The interface definition is as follows:

```python
class ModelGroup
    def __init__(self, flags=ModelGroupFlag.SHARE_WORKSPACE)
    def add_model(self, models)
    def cal_max_size_of_workspace(self, model_type, context)
```

```C++
// class ModelGroup
ModelGroup(ModelGroupFlag flags = ModelGroupFlag::kShareWorkspace);
Status AddModel(const std::vector<std::string> &model_path_list);
Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list);
Status AddModel(const std::vector &model_list);
Status AddModel(const std::vector &model_list);
```

## MindSpore 2.0.0 Release Notes

### Major Features and Improvements

#### PyNative

- [STABLE] Dynamic shape is fully supported on framework. For detailed operator support, refer to [Dynamic Shape Support Status of nn Interface](https://www.mindspore.cn/docs/en/r2.0/note/dynamic_shape_nn.html), [Dynamic Shape Support Status of ops Interface](https://www.mindspore.cn/docs/en/r2.0/note/dynamic_shape_func.html), and [Dynamic Shape Support Status of primitive Interface](https://www.mindspore.cn/docs/en/r2.0/note/dynamic_shape_primitive.html).

#### AutoParallel

- [STABLE] Build new MindFormers independent repositpry, providing distributed parallel suite, replacing mindspore.nn.transformer module.
- [DEMO] Distributed parallel operator Gather supports the BatchDim attribute.
- [DEMO] Streamline parallel supports specifying any dimension of the input data as the Batch dimension.

### API Change

#### operator

- Add operator primitive for `mindspore.ops.AdaptiveAvgPool2D` .
- Add operator primitive for `mindspore.ops.BatchToSpaceNDV2` .
- Add operator primitive for `mindspore.ops.CeLU` .
- Add operator primitive for `mindspore.ops.ExtractVolumePatches` .
- Add operator primitive for `mindspore.ops.FFTWithSize` .
- Add operator primitive for `mindspore.ops.FillDiagonal` .
- Add operator primitive for `mindspore.ops.FractionalMaxPool3DWithFixedKsize` .
- Add operator primitive for `mindspore.ops.Im2Col` .
- Add operator primitive for `mindspore.ops.MaskedScatter` .
- Add operator primitive for `mindspore.ops.MatrixBandPart` .
- Add operator primitive for `mindspore.ops.MatrixInverse` .
- Add operator primitive for `mindspore.ops.MaxPoolWithArgmaxV2` .
- Add operator primitive for `mindspore.ops.Ormqr` .
- Add operator primitive for `mindspore.ops.RandpermV2` .
- Add operator primitive for `mindspore.ops.ResizeBicubic` .
- Add operator primitive for `mindspore.ops.Triu` .
- Add operator primitive for `mindspore.ops.Zeta` .

#### Backwards Incompatible Change

- Interface: mindspore.ops.MultitypeFuncGraph

  Change: The interface parameter doc_url is used as a test feature in MindSpore 2.0.0.rc1 version. After the optimization of MindSpore 2.0.0 version, users do not need to configure this parameter, so this parameter is deleted in MindSpore 2.0.0 version.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.ops.MultitypeFuncGraph（name, read_value=False, doc_url=""）
  </pre>
  </td>
  <td><pre>
  mindspore.ops.MultitypeFuncGraph（name, read_value=False）
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.set_context(auto_tune_mode="GA,RL")

  Change: The AutoTune tool has been deprecated, delete auto_tune_mode option, new tuning tools will be planned in the future.

- Interface: mindspore.set_context(mode=PYNATIVE_MODE)

  Change: The default value is changed from GRAPH_MODE to PYNATIVE_MODE.

  Description: If the running mode is not set and the diagram mode needs to be set, use the following method:
  mindspore.set_context(mode=GRAPH_MODE).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.set_context(mode=GRAPH_MODE)
  </pre>
  </td>
  <td><pre>
  mindspore.set_context(mode=PYNATIVE_MODE)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.train.Model.train

  Change: The default value of dataset_sink_mode is changed from True to False.

  Description: If dataset_sink_mode is not set and the data sinking mode needs to be set, use the following method:
  Model.train(dataset_sink_mode=True).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Model.train(dataset_sink_mode=True)
  </pre>
  </td>
  <td><pre>
  Model.train(dataset_sink_mode=False)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.export

  Change: The file_format parameter is changed from AIR to no default value.

  Description: If file_format is not set in the original mode, you need to set file_format additionally. In this case, use the following method:
  mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.export(net, *inputs, file_name,
                   file_format="AIR", **kwargs)
  </pre>
  </td>
  <td><pre>
  mindspore.export(net, *inputs, file_name,
                   file_format, **kwargs)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.norm

  Change: The ord parameter function is extended to support multiple forms.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)
  >>> # Example:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, [0, 1], p=2)
  </pre></td>
  <td><pre>
  ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)
  >>> # Example:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, ord=2, dim=(0, 1))
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.Tensor.norm

  Change: The ord parameter function is extended to support multiple forms.

  Description: For details, see the example of ops.norm.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Tensor.norm(axis, p=2, keep_dims=False, epsilon=1e-12)
  </pre>
  </td>
  <td><pre>
  Tensor.norm(ord=None, dim=None, keepdim=False, *, dtype=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout

  Change: The seed0 and seed1 parameters are deleted and seed=None parameter is added. Instead of returning Tensors and masks, only Tensors are returned. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout(x, p=0.5, seed0=0, seed1=0)
  >>> # Example:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output, mask = dropout(x, p=0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout(input, p=0.5, training=True, seed=None)
  >>> # Example:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output = ops.dropout(input, p=0.5，training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout2d

  Change: Return value is changed from Tensor and mask to Tensor only. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout2d(x, p=0.5)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout2d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout2d(input, p=0.5, training=True)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout2d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout3d

  Change: Return value is changed from Tensor and mask to Tensor only. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout3d(x, p=0.5)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout3d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout3d(input, p=0.5, training=True)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout3d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.std

  Change: The interface is reconstructed, and the interface usage mode is more consistent with user habits.

  Description: If parameter `unbiased` has been set, use the following alternative: `unbiased=False` -> `ddof=0`, `unbiased=True` -> `ddof=1`.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.std(input_x, axis=(), unbiased=True, keep_dims=False)
  </pre>
  </td>
  <td><pre>
  ops.std(input, axis=None, ddof=0, keepdims=False)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.load_param_into_net

  Change: Parameters that are not loaded in the ckpt are added as return values.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  net_param = load_param_into_net()
  </pre>
  </td>
  <td><pre>
  net_param, ckpt_param = load_param_into_net()
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.nn.BCELoss

  Change: The default value of `reduction` is changed from 'none' to 'mean'.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  BCELoss(weight=None, reduction='none')
  >>> # Example:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight, reduction='mean')
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </pre>
  </td>
  <td><pre>
  BCELoss(weight=None, reduction='mean')
  >>> # Example:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight)
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.split

  Change: The interface is reconstructed. The interface usage mode is more suitable for users. The sequence of the second and third parameters is adjusted, and the split_size_or_sections function is modified and extended.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.split(input_x, axis=0, output_num=1)
  >>> # Example:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, axis=1, output_num=4)
  </pre>
  </td>
  <td><pre>
  ops.split(tensor, split_size_or_sections, axis=0)
  >>> # Example:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, split_size_or_sections=1, axis=1)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.Tensor.split

  Change: The interface is reconstructed. The interface usage mode is more suitable for users. The positions of the two parameters is adjusted, and the split_size_or_sections function is modified and extended.

  Description: For details, see the example of ops.split.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Tensor.split(axis=0, output_num=1)
  </pre>
  </td>
  <td><pre>
  Tensor.split(split_size_or_sections, axis=0)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.pad

  Change: Modify the parameter name paddings to padding, and the mode and value functions are added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.pad(input_x, paddings)
  >>> # Example:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = ((1, 2), (2, 1))
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  <td><pre>
  ops.pad(input_x, padding, mode='constant', value=None)
  >>> # Example:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = (2, 1, 1, 2)
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.meshgrid

  Change: The input parameter is changed from `inputs` to `*input`.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.meshgrid(inputs, indexing='xy')
  >>> # Example:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  output = ops.meshgrid((x, y, z), indexing='xy')
  </pre>
  </td>
  <td><pre>
  ops.meshgrid(*inputs, indexing='xy')
  >>> # Example:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  output = ops.meshgrid(x, y, z, indexing='xy')
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.max

  Change: Return value exchange sequence. The value is changed from "index, value" to "value, index".

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.max(x, axis=0, keep_dims=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.max(input)
  >>> print(index, output)
  >>> 3 0.7
  </pre>
  </td>
  <td><pre>
  ops.max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.max(input, axis=0)
  >>> print(output, index)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.min

  Change: Return value exchange sequence. The value is changed from "index, value" to "value, index".

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.min(x, axis=0, keep_dims=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.min(input)
  >>> 0 0.0
  </pre>
  </td>
  <td><pre>
  ops.min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.min(input, keepdims=True)
  >>> 0.0 0
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.random_gamma

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.random_gamma(shape, alpha, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.random_gamma(shape, alpha, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.standard_laplace

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.standard_laplace(shape, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.standard_laplace(shape, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.standard_normal

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.standard_normal(shape, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.standard_normal(shape, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.bernoulli

  Change: The default value of seed is changed from -1 to None. Meets the actual application scenario.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.bernoulli(x, p=0.5, seed=-1)
  </pre>
  </td>
  <td><pre>
  ops.bernoulli(input, p=0.5, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.data_sink

  Change: Deleted the steps parameter. Parameter name jit is changed to jit_config, and new input_signature parameter is added. The usability is improved to meet the requirements of actual application scenarios.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.data_sink(fn, dataset, steps,
                      sink_size=1, jit=False)
  </pre>
  </td>
  <td><pre>
  mindspore.data_sink(fn, dataset, sink_size=1,
                      jit_config=None, input_signature=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.conv2d

  Change: Extend Interface Function. Add the bias parameter and modify the parameter name and parameter sequence.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  conv2d(inputs, weight, pad_mode="valid",
         padding=0, stride=1, dilation=1, group=1)
  </pre>
  </td>
  <td><pre>
  conv2d(input, weight, bias=None, stride=1,
         pad_mode="valid", padding=0, dilation=1, groups=1)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.vision.Pad

  Change: Adjust the input parameter padding of Pad, RandomCrop, and RandomCropWithBbox. When the input length of Padding is 2, the first value is used to fill the left/upper boundary, the second value is used to fill the right/lower boundary, and the first value is used to fill the left/right boundary. Fill the upper/lower boundary with the second value.

  Description: The padding parameter whose size is 2 is not compatible with the effect of the earlier version. The padding parameter needs to be explicitly represented (left, right, top, and bottom).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2))
  Indicates that the left/upper part of the image is filled with 1 pixel,
  and the right/down part is filled with 2 pixels.
  </pre>
  </td>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2,1,2))
  Indicates that the left/upper part of the image is filled with 1 pixel,
  and the right/down part is filled with 2 pixels.
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.map

  Change: Delete the column_order parameter. In most cases, output_columns and column_order have the same value. Therefore, column_order does not need to be transferred. To adjust the sequence of data columns, use mindspore.dataset.Dataset.project.

  Description:

  1. If the column sequence does not need to be changed, delete the column_order parameter.
  2. If you need to specify the data column sequence, delete the column_order parameter and add a project method to the end of the parameter for column transformation (as in the following example).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"],
  ...                       column_order=["column_c", "column_b"])
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.batch

  Change: Delete the column_order parameter. In most cases, output_columns and column_order have the same value. Therefore, column_order does not need to be transferred. To adjust the sequence of data columns, use mindspore.dataset.Dataset.project.

  Description:

  1. If the column sequence does not need to be changed, delete the column_order parameter.
  2. If you need to specify the data column sequence, delete the column_order parameter and add a project method to the end of the parameter for column transformation (as in the following example).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4,
  ...                         input_columns=["column_a"],
  ...                         output_columns=["column_b", "column_c"],
  ...                         column_order=["column_c", "column_b"])
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4, input_columns=["column_a"]
  ...                         output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.batch

  Change: Split the batch method into two methods: batch and padded_batch. The pad_info parameter is moved from the batch method to the padded_batch method.

  Description: To use the pad_info parameter, use the padded_batch method instead.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4,
  ...                         drop_remainder=True, pad_info=...)
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.padded_batch(batch_size=4,
  ...                                drop_remainder=True, pad_info=...)
  </pre>
  </td>
  </tr>
  </table>

### Bug fixes

- [I62I3J] fix inference failure of BGCF network on Ascend 310
- [I7C2W3] fix error issuse of null pointer when enabling multiple loss in parallel pipeline scenarios

### Contributors

Thanks goes to these wonderful people:

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

Contributions of any kind are welcome!

## MindSpore 2.0.0-rc1 Release Notes

### Major Features and Improvements

#### FrontEnd

- [BETA] Statement with "return", "return None" and with no return of function are supported in `GRAPH_MODE`.
- [BETA] Object with `list` type are supported in `GRAPH_MODE`.
- [BETA] Statement with "raise" are supported in variable condition situation in `GRAPH_MODE`.
- [STABLE] Functional call supports data sinking mode.
- [BETA] The Transformer layer in nn module is added to provide easy-to-use Transformer APIs. Batch_size does not need to be defined. Dynamic seq_length is supported.

#### DataSet

- [STABLE] In the Ascend environment，the timeout waiting time in data sink mode is adjusted to 1900s by default. This solves the problem that the GetNext operator may time out due to environment resource competition and large computing workload in data sinking mode.
- [STABLE] MindRecord supports to query the schemas and number samples. MindRecord provides multi-process writing mode, allowing users to generate MindRecord data files in parallel.
- [STABLE] The Dataset pipeline can process any Python object. For details, see [Supporting Python Objects in Dataset Pipeline](https://www.mindspore.cn/tutorials/en/r2.0/advanced/dataset/python_objects.html).

#### AutoParallel

- [STABLE] The strategies of whole parameters can be saved when saving strategy.
- [STABLE] The Conv3D/MaxPool3D/AvgPool3D distributed operator is supported.
- [STABLE] Support operator-level parallelism and optimizer-level parallelism under the PyNative with shard: parallel training and the Model API are decoupled to provide basic parallel expression capabilities.
- [STABLE] Support operator-level parallelism, and optimizer-level parallelism under the Graph mode: parallel training and the Model API are decoupled to provide basic parallel expression capabilities.
- [BETA] Supports customized distributed graph segmentation, improving the flexibility of distributed training.

#### Runtime

- [STABLE] Control flow supports subgraph sink.
- [STABLE] Support CUDA 11.6.
- [STABLE] Support for operator selection and execution of List/Tuple/Scalar type kernel to match native Python expression.
- [STABLE] Kernel that is not supported by hardware can automatically select CPU kernel.
- [STABLE] Support heterogeneous execution within subgraph.

#### Ascend

- [STABLE] Support overflow detection scheme and HCCL runtime overflow check.
- [STABLE] Support dump of communication operators.

#### Profiler

- [STABLE] Rich Profiler collection item configuration, users can collect performance data in more detail.

#### Dump

- [BETA] Single card in PyNatvie mode supports operator overflow detection.
- [BETA] Graph mode supports hccl operator dump.

### API Change

- [STABLE] Add computing APIs, such as MaxUnpool, ReplicationPad, and GaussianNLLLoss.
  For details, visit <https://www.mindspore.cn/docs/en/r2.0/api_python/mindspore.html>.
- [STABLE] Extend inventory API functions, such as AvgPool, pad, norm, and interplate.

#### operator

- [BETA] Add operator primitive for `mindspore.ops.AdaptiveAvgPool3D`.
- [BETA] Add operator primitive for `mindspore.ops.AffineGrid`.
- [BETA] Add operator primitive for `mindspore.ops.Angle`.
- [BETA] Add operator primitive for `mindspore.ops.BartlettWindow`.
- [BETA] Add operator primitive for `mindspore.ops.Bernoulli`.
- [BETA] Add operator primitive for `mindspore.ops.BesselI0`.
- [BETA] Add operator primitive for `mindspore.ops.BesselI1`.
- [BETA] Add operator primitive for `mindspore.ops.BesselJ0`.
- [BETA] Add operator primitive for `mindspore.ops.BesselJ1`.
- [BETA] Add operator primitive for `mindspore.ops.BesselK0`.
- [BETA] Add operator primitive for `mindspore.ops.BesselK0e`.
- [BETA] Add operator primitive for `mindspore.ops.BesselK1`.
- [BETA] Add operator primitive for `mindspore.ops.BesselK1e`.
- [BETA] Add operator primitive for `mindspore.ops.BesselY0`.
- [BETA] Add operator primitive for `mindspore.ops.BesselY1`.
- [BETA] Add operator primitive for `mindspore.ops.Bincount`.
- [BETA] Add operator primitive for `mindspore.ops.BlackmanWindow`.
- [BETA] Add operator primitive for `mindspore.ops.ChannelShuffle`.
- [BETA] Add operator primitive for `mindspore.ops.Cholesky`.
- [BETA] Add operator primitive for `mindspore.ops.Col2Im`.
- [BETA] Add operator primitive for `mindspore.ops.Complex`.
- [BETA] Add operator primitive for `mindspore.ops.ComplexAbs`.
- [BETA] Add operator primitive for `mindspore.ops.Cross`.
- [BETA] Add operator primitive for `mindspore.ops.CTCLossV2`.
- [BETA] Add operator primitive for `mindspore.ops.Cummin`.
- [BETA] Add operator primitive for `mindspore.ops.Diag`.
- [BETA] Add operator primitive for `mindspore.ops.Digamma`.
- [BETA] Add operator primitive for `mindspore.ops.Expand`.
- [BETA] Add operator primitive for `mindspore.ops.Fmax`.
- [BETA] Add operator primitive for `mindspore.ops.Gcd`.
- [BETA] Add operator primitive for `mindspore.ops.Geqrf`.
- [BETA] Add operator primitive for `mindspore.ops.GLU`.
- [BETA] Add operator primitive for `mindspore.ops.GridSampler2D`.
- [BETA] Add operator primitive for `mindspore.ops.GridSampler3D`.
- [BETA] Add operator primitive for `mindspore.ops.HammingWindow`.
- [BETA] Add operator primitive for `mindspore.ops.Heaviside`.
- [BETA] Add operator primitive for `mindspore.ops.Hypot`.
- [BETA] Add operator primitive for `mindspore.ops.Igamma`.
- [BETA] Add operator primitive for `mindspore.ops.IndexFill`.
- [BETA] Add operator primitive for `mindspore.ops.InplaceIndexAdd`.
- [BETA] Add operator primitive for `mindspore.ops.InplaceUpdateV2`.
- [BETA] Add operator primitive for `mindspore.ops.Lcm`.
- [BETA] Add operator primitive for `mindspore.ops.LeftShift`.
- [BETA] Add operator primitive for `mindspore.ops.LogicalXor`.
- [BETA] Add operator primitive for `mindspore.ops.Logit`.
- [BETA] Add operator primitive for `mindspore.ops.LogSpace`.
- [BETA] Add operator primitive for `mindspore.ops.LuUnpack`.
- [BETA] Add operator primitive for `mindspore.ops.MatrixDiagPartV3`.
- [BETA] Add operator primitive for `mindspore.ops.MatrixDiagV3`.
- [BETA] Add operator primitive for `mindspore.ops.MatrixSetDiagV3`.
- [BETA] Add operator primitive for `mindspore.ops.MaxPool3DWithArgmax`.
- [BETA] Add operator primitive for `mindspore.ops.MaxUnpool2D`.
- [BETA] Add operator primitive for `mindspore.ops.MaxUnpool3D`.
- [BETA] Add operator primitive for `mindspore.ops.MultiMarginLoss`.
- [BETA] Add operator primitive for `mindspore.ops.MultinomialWithReplacement`.
- [BETA] Add operator primitive for `mindspore.ops.Mvlgamma`.
- [BETA] Add operator primitive for `mindspore.ops.NanToNum`.
- [BETA] Add operator primitive for `mindspore.ops.NextAfter`.
- [BETA] Add operator primitive for `mindspore.ops.Orgqr`.
- [BETA] Add operator primitive for `mindspore.ops.Polygamma`.
- [BETA] Add operator primitive for `mindspore.ops.ResizeBilinearV2`.
- [BETA] Add operator primitive for `mindspore.ops.RightShift`.
- [BETA] Add operator primitive for `mindspore.ops.ScatterNdDiv`.
- [BETA] Add operator primitive for `mindspore.ops.ScatterNdMul`.
- [BETA] Add operator primitive for `mindspore.ops.SearchSorted`.
- [BETA] Add operator primitive for `mindspore.ops.Sinc`.
- [BETA] Add operator primitive for `mindspore.ops.Trace`.
- [BETA] Add operator primitive for `mindspore.ops.Tril`.
- [BETA] Add operator primitive for `mindspore.ops.TrilIndices`.
- [BETA] Add operator primitive for `mindspore.ops.TriuIndices`.
- [BETA] Add operator primitive for `mindspore.ops.UniqueConsecutive`.
- [STABLE] Add operator primitive for `mindspore.ops.Cummax`.
- [STABLE] Add operator primitive for `mindspore.ops.FillV2`.
- [STABLE] Add operator primitive for `mindspore.ops.IsClose`.
- [STABLE] Add operator primitive for `mindspore.ops.MatrixSolve`.
- [STABLE] Add operator primitive for `mindspore.ops.Median`.
- [STABLE] Add operator primitive for `mindspore.ops.MultilabelMarginLoss`.
- [STABLE] Add operator primitive for `mindspore.ops.NonZero`.
- [STABLE] Add operator primitive for `mindspore.ops.Pdist`.
- [STABLE] Add operator primitive for `mindspore.ops.Polar`.
- [STABLE] Add operator primitive for `mindspore.ops.RandomGamma`.
- [STABLE] Add operator primitive for `mindspore.ops.RandomPoisson`.
- [STABLE] Add operator primitive for `mindspore.ops.RandomShuffle`.
- [STABLE] Add operator primitive for `mindspore.ops.Renorm`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterNdMax`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterNdMin`.
- [STABLE] Add operator primitive for `mindspore.ops.Svd`.
- [STABLE] Add operator primitive for `mindspore.ops.TripletMarginLoss`.

#### Deleted APIs

- The `mindspore.compression` feature was deprecated at MindSpore 1.8 and is removed in this version.
  The following `mindspore.nn.quant` interfaces are also removed simultaneously: `mindspore.nn.FakeQuantWithMinMaxObserver`, `mindspore.nn.Conv2dBnFoldQuantOneConv`, `mindspore.nn.Conv2dBnFoldQuant`, `mindspore.nn.Conv2dBnWithoutFoldQuant`, `mindspore.nn.Conv2dQuant`, `mindspore.nn.DenseQuant`, `mindspore.nn.ActQuant`, `mindspore.nn.TensorAddQuant`, `mindspore.nn.ActQuant`, `mindspore.nn.MulQuant`. Please use [MindSpore Golden Stick](https://gitee.com/mindspore/golden-stick) instead to implement QuantAwareTraining in MindSpore.
- The `mindspore.dataset.close_pool`, `mindspore.dataset.to_device`, and `mindspore.dataset.set_dynamic_columns` interfaces are discarded in earlier version and being removed in this version.

#### Backwards Incompatible Change

- Interface: mindspore.set_context(mode=PYNATIVE_MODE)

  Change: The default value is changed from GRAPH_MODE to PYNATIVE_MODE.

  Description: If the running mode is not set and the diagram mode needs to be set, use the following method:
  mindspore.set_context(mode=GRAPH_MODE).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.set_context(mode=GRAPH_MODE)
  </pre>
  </td>
  <td><pre>
  mindspore.set_context(mode=PYNATIVE_MODE)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.train.Model.train

  Change: The default value of dataset_sink_mode is changed from True to False.

  Description: If dataset_sink_mode is not set and the data sinking mode needs to be set, use the following method:
  Model.train(dataset_sink_mode=True).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Model.train(dataset_sink_mode=True)
  </pre>
  </td>
  <td><pre>
  Model.train(dataset_sink_mode=False)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.export

  Change: The file_format parameter is changed from AIR to no default value.

  Description: If file_format is not set in the original mode, you need to set file_format additionally. In this case, use the following method:
  mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.export(net, *inputs, file_name,
                   file_format="AIR", **kwargs)
  </pre>
  </td>
  <td><pre>
  mindspore.export(net, *inputs, file_name,
                   file_format, **kwargs)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.norm

  Change: The ord parameter function is extended to support multiple forms.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)
  >>> # Example:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, [0, 1], p=2)
  </pre></td>
  <td><pre>
  ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)
  >>> # Example:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, ord=2, dim=(0, 1))
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.Tensor.norm

  Change: The ord parameter function is extended to support multiple forms.

  Description: For details, see the example of ops.norm.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Tensor.norm(axis, p=2, keep_dims=False, epsilon=1e-12)
  </pre>
  </td>
  <td><pre>
  Tensor.norm(ord=None, dim=None, keepdim=False, *, dtype=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout

  Change: The seed0 and seed1 parameters are deleted and seed=None parameter is added. Instead of returning Tensors and masks, only Tensors are returned. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout(x, p=0.5, seed0=0, seed1=0)
  >>> # Example:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output, mask = dropout(x, p=0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout(input, p=0.5, training=True, seed=None)
  >>> # Example:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output = ops.dropout(input, p=0.5，training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout2d

  Change: Return value is changed from Tensor and mask to Tensor only. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout2d(x, p=0.5)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout2d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout2d(input, p=0.5, training=True)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout2d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.dropout3d

  Change: Return value is changed from Tensor and mask to Tensor only. The input parameter training=True is added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout3d(x, p=0.5)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout3d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout3d(input, p=0.5, training=True)
  >>> # Example:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout3d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.std

  Change: The interface is reconstructed, and the interface usage mode is more consistent with user habits.

  Description: If parameter `unbiased` has been set, use the following alternative: `unbiased=False` -> `ddof=0`, `unbiased=True` -> `ddof=1`.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.std(input_x, axis=(), unbiased=True, keep_dims=False)
  </pre>
  </td>
  <td><pre>
  ops.std(input, axis=None, ddof=0, keepdims=False)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.load_param_into_net

  Change: Parameters that are not loaded in the ckpt are added as return values.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  net_param = load_param_into_net()
  </pre>
  </td>
  <td><pre>
  net_param, ckpt_param = load_param_into_net()
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.nn.BCELoss

  Change: The default value of `reduction` is changed from 'none' to 'mean'.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  BCELoss(weight=None, reduction='none')
  >>> # Example:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight, reduction='mean')
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </pre>
  </td>
  <td><pre>
  BCELoss(weight=None, reduction='mean')
  >>> # Example:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight)
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.split

  Change: The interface is reconstructed. The interface usage mode is more suitable for users. The sequence of the second and third parameters is adjusted, and the split_size_or_sections function is modified and extended.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.split(input_x, axis=0, output_num=1)
  >>> # Example:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, axis=1, output_num=4)
  </pre>
  </td>
  <td><pre>
  ops.split(tensor, split_size_or_sections, axis=0)
  >>> # Example:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, split_size_or_sections=1, axis=1)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.Tensor.split

  Change: The interface is reconstructed. The interface usage mode is more suitable for users. The positions of the two parameters is adjusted, and the split_size_or_sections function is modified and extended.

  Description: For details, see the example of ops.split.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  Tensor.split(axis=0, output_num=1)
  </pre>
  </td>
  <td><pre>
  Tensor.split(split_size_or_sections, axis=0)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.pad

  Change: Modify the parameter name paddings to padding, and the mode and value functions are added.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.pad(input_x, paddings)
  >>> # Example:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = ((1, 2), (2, 1))
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  <td><pre>
  ops.pad(input_x, padding, mode='constant', value=None)
  >>> # Example:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = (2, 1, 1, 2)
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.meshgrid

  Change: The input parameter is changed from `inputs` to `*input`.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.meshgrid(inputs, indexing='xy')
  >>> # Example:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  output = ops.meshgrid((x, y, z), indexing='xy')
  </pre>
  </td>
  <td><pre>
  ops.meshgrid(*inputs, indexing='xy')
  >>> # Example:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  output = ops.meshgrid(x, y, z, indexing='xy')
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.max

  Change: Return value exchange sequence. The value is changed from "index, value" to "value, index".

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.max(x, axis=0, keep_dims=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.max(input)
  >>> print(index, output)
  >>> 3 0.7
  </pre>
  </td>
  <td><pre>
  ops.max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.max(input, axis=0)
  >>> print(output, index)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.min

  Change: Return value exchange sequence. The value is changed from "index, value" to "value, index".

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.min(x, axis=0, keep_dims=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.min(input)
  >>> 0 0.0
  </pre>
  </td>
  <td><pre>
  ops.min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # Example:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.min(input, keepdims=True)
  >>> 0.0 0
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.random_gamma

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.random_gamma(shape, alpha, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.random_gamma(shape, alpha, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.standard_laplace

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.standard_laplace(shape, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.standard_laplace(shape, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.standard_normal

  Change: The seed2 parameter is deleted and seed=0 is changed to None. The framework behavior is unified and complies with the actual application scenarios and habits of users.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.standard_normal(shape, seed=0, seed2=0)
  </pre>
  </td>
  <td><pre>
  ops.standard_normal(shape, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.bernoulli

  Change: The default value of seed is changed from -1 to None. Meets the actual application scenario.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  ops.bernoulli(x, p=0.5, seed=-1)
  </pre>
  </td>
  <td><pre>
  ops.bernoulli(input, p=0.5, seed=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.data_sink

  Change: Deleted the steps parameter. Parameter name jit is changed to jit_config, and new input_signature parameter is added. The usability is improved to meet the requirements of actual application scenarios.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.data_sink(fn, dataset, steps,
                      sink_size=1, jit=False)
  </pre>
  </td>
  <td><pre>
  mindspore.data_sink(fn, dataset, sink_size=1,
                      jit_config=None, input_signature=None)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.ops.conv2d

  Change: Extend Interface Function. Add the bias parameter and modify the parameter name and parameter sequence.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  conv2d(inputs, weight, pad_mode="valid",
         padding=0, stride=1, dilation=1, group=1)
  </pre>
  </td>
  <td><pre>
  conv2d(input, weight, bias=None, stride=1,
         pad_mode="valid", padding=0, dilation=1, groups=1)
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.vision.Pad

  Change: Adjust the input parameter padding of Pad, RandomCrop, and RandomCropWithBbox. When the input length of Padding is 2, the first value is used to fill the left/upper boundary, the second value is used to fill the right/lower boundary, and the first value is used to fill the left/right boundary. Fill the upper/lower boundary with the second value.

  Description: The padding parameter whose size is 2 is not compatible with the effect of the earlier version. The padding parameter needs to be explicitly represented (left, right, top, and bottom).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2))
  Indicates that the left/upper part of the image is filled with 1 pixel,
  and the right/down part is filled with 2 pixels.
  </pre>
  </td>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2,1,2))
  Indicates that the left/upper part of the image is filled with 1 pixel,
  and the right/down part is filled with 2 pixels.
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.map

  Change: Delete the column_order parameter. In most cases, output_columns and column_order have the same value. Therefore, column_order does not need to be transferred. To adjust the sequence of data columns, use mindspore.dataset.Dataset.project.

  Description:

  1. If the column sequence does not need to be changed, delete the column_order parameter.
  2. If you need to specify the data column sequence, delete the column_order parameter and add a project method to the end of the parameter for column transformation (as in the following example).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"],
  ...                       column_order=["column_c", "column_b"])
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.batch

  Change: Delete the column_order parameter. In most cases, output_columns and column_order have the same value. Therefore, column_order does not need to be transferred. To adjust the sequence of data columns, use mindspore.dataset.Dataset.project.

  Description:

  1. If the column sequence does not need to be changed, delete the column_order parameter.
  2. If you need to specify the data column sequence, delete the column_order parameter and add a project method to the end of the parameter for column transformation (as in the following example).

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4,
  ...                         input_columns=["column_a"],
  ...                         output_columns=["column_b", "column_c"],
  ...                         column_order=["column_c", "column_b"])
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4, input_columns=["column_a"]
  ...                         output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </pre>
  </td>
  </tr>
  </table>

- Interface: mindspore.dataset.Dataset.batch

  Change: Split the batch method into two methods: batch and padded_batch. The pad_info parameter is moved from the batch method to the padded_batch method.

  Description: To use the pad_info parameter, use the padded_batch method instead.

  <table>
  <tr>
  <td style="text-align:center"> Original Interface </td> <td style="text-align:center"> Interface v2.0.0-rc1 </td>
  </tr>
  <tr>
  <td><pre>
  >>> dataset = dataset.batch(batch_size=4,
  ...                         drop_remainder=True, pad_info=...)
  </pre>
  </td>
  <td><pre>
  >>> dataset = dataset.padded_batch(batch_size=4,
  ...                                drop_remainder=True, pad_info=...)
  </pre>
  </td>
  </tr>
  </table>

### Bug fixes

- [I66PE6] fix AssignSub primitive abnormal input leads to coredump.

- [I6F5E6] fix data_sink function timeout on Ascend.

### Others

- Windows support is still being optimized,this version does not support now.It will be available for download in version 2.0.

### Contributors

Thanks goes to these wonderful people:

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

Contributions of any kind are welcome!

## MindSpore Lite 2.0.0-rc1 Release Notes

### Major Features and Improvements

#### MindSpore Lite Cloud Inference

The original MindSpore Lite is mainly used for edge devices such as mobile phones and head units. Cloud inference is added to support scenarios with multiple backend hardware resources on the cloud, supports Ascend and NVIDIA GPU inference cards, and efficiently utilizes multi-core resources on the cloud.

The original cloud inference integrated through MindSpore training can be changed to MindSpore Lite. For details, see [Quick Start to Cloud-side Inference](https://mindspore.cn/lite/docs/en/r2.0/quick_start/one_hour_introduction_cloud.html). To retain the original integration method, see [Inference](https://mindspore.cn/docs/en/r2.0/faq/inference.html).

- [STABLE] Support MindIR model files.
- [STABLE] Third-party Onnx, TensorFlow, and Caffe models can be converted to MindIR model files using the MindSpore Lite conversion tool.
- [STABLE] One release package supports multiple hardware backends: Ascend, NVIDIA GPU, CPU.
- [STABLE] Supports the `Model` interface and `ModelParallelRunner` concurrent inference interface.
- [STABLE] Supports C++, Python, and Java inference interfaces.

#### API

- Due to the defects of the original Python API that many configuration parameters and complex usage, the usability of The Python APIs are optimized in version 2.0. The optimizations include class construction methods and class attribute adjustment. In addition, the Python APIs in version 2.0 and later will be integrated into the cloud-side inference scenario, which are incompatible with Python APIs of the earlier versions. For details, see [Python API](https://www.mindspore.cn/lite/api/en/r2.0/mindspore_lite.html).

## MindSpore 2.0.0-alpha Release Notes

### Major Features and Improvements

#### PyNative

- The default mode of MindSpore is switched to PyNative. If you want to manually set the mode, please refer to [Computational Graph](https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/compute_graph.html).
- Support dynamic shape without padding, three networks are supported as demos: Transformer-GPU, YOLOV5-GPU, ASR-Ascend. Transformer-GPU and YOLOV5-GPU can be downloaded from [models](https://gitee.com/mindspore/models/tree/dynamic_shape). Only the following operators are available on Ascend backend: Add、Assign、BatchMatMul、BiasAdd、BiasAddGrad、Cast、Conv2D、Conv2DBackpropFilter、Conv2DBackpropInput、CTCLoss、Div、Dropout、DropoutDoMask、Equal、ExpandDims、Gather、GetNext、LayerNorm、LayerNormGrad、LessEqual、Load、Log、LogicalAnd、LogicalNot、LogicalOr、LogSoftmax、LogSoftmaxGrad、MatMul、Maximum、Mul、Neg、NotEqual、NPUAllocFloatStatus、NPUClearFloatStatus、OneHot、RealDiv、Reciprocal、ReduceMean、ReduceSum、ReLU、ReluGrad、Reshape、Select、Softmax、StridedSlice、Sub、Tile、Transpose、UnsortedSegmentSum、ZerosLike. The remaining operators have not been fully verified, please use them as appropriate.

#### DataSet

- The TFRecordDataset API can directly read TFRecord files compressed by GZIP or ZLIB.
- The NumpySlicesDataset API can process data of different dimensions at the same time.
- Optimize the structure of error log to display more clear call stack information for debugging.
- Fixed `mindspore.dataset.config.set_seed` does not take effect for random seeds in distributed training scenarios.

#### AutoParallel

- Supports more operators with distributed implements.

  Element Wise Operators:AddN, BitwiseAnd, BitwiseOr, BitwiseXor, CumProd, HShrink, HSigmoid, IsFinite, Mish, MulNoNan, Rint, SeLU, SoftShrink, TruncateDiv, TruncateMod, Xdivy Xlogy, InplaceAdd, InplacSub, InplaceUpdate, Cdist, L2Loss, Lerp.

  Math Operators:SquaredDifference, Erfinv, MaskedFill, SplitV, Gamma, KLDivLoss, LinSpace.

  Scatter Operators:ScatterAdd,ScatterDiv,ScatterMax,ScatterMul,ScatterNdAdd,ScatterNdSub,ScatterNdUpdate,ScatterSub,TensorScatterAdd,TensorScatterDiv,TensorScatterMax,TensorScatterMax,TensorScatterMul,TensorScatterAdd,TensorScatterUpdate.

- Add new apis `transform_checkpoints` and `transform_checkpoint_by_rank` to transfer the distributed checkpoint files by strategy files. Please refer to [Distributed Resilience Training and Inference](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/resilience_train_and_predict.html).

### API Change

#### operator

- [STABLE] Add operator primitive for `mindspore.ops.AdaptiveMaxPool3D`.
- [STABLE] Add operator primitive for `mindspore.ops.AdjustHue`.
- [STABLE] Add operator primitive for `mindspore.ops.BartlettWindow`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselJ0`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselJ1`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselK0`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselK0e`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselK1`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselK1e`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselY0`.
- [STABLE] Add operator primitive for `mindspore.ops.BesselY1`.
- [STABLE] Add operator primitive for `mindspore.ops.Betainc`.
- [STABLE] Add operator primitive for `mindspore.ops.Bincount`.
- [STABLE] Add operator primitive for `mindspore.ops.BlackmanWindow`.
- [STABLE] Add operator primitive for `mindspore.ops.Bucketize`.
- [STABLE] Add operator primitive for `mindspore.ops.CombinedNonMaxSuppression`.
- [STABLE] Add operator primitive for `mindspore.ops.CompareAndBitpack`.
- [STABLE] Add operator primitive for `mindspore.ops.Complex`.
- [STABLE] Add operator primitive for `mindspore.ops.DataFormatVecPermute`.
- [STABLE] Add operator primitive for `mindspore.ops.EuclideanNorm`.
- [STABLE] Add operator primitive for `mindspore.ops.Expand`.
- [STABLE] Add operator primitive for `mindspore.ops.ExtractGlimpse`.
- [STABLE] Add operator primitive for `mindspore.ops.FillDiagonal`.
- [STABLE] Add operator primitive for `mindspore.ops.FractionalAvgPool`.
- [STABLE] Add operator primitive for `mindspore.ops.FractionalMaxPool`.
- [STABLE] Add operator primitive for `mindspore.ops.Gcd`.
- [STABLE] Add operator primitive for `mindspore.ops.HammingWindow`.
- [STABLE] Add operator primitive for `mindspore.ops.Histogram`.
- [STABLE] Add operator primitive for `mindspore.ops.HSVToRGB`.
- [STABLE] Add operator primitive for `mindspore.ops.Lcm`.
- [STABLE] Add operator primitive for `mindspore.ops.LeftShift`.
- [STABLE] Add operator primitive for `mindspore.ops.ListDiff`.
- [STABLE] Add operator primitive for `mindspore.ops.LogSpace`.
- [STABLE] Add operator primitive for `mindspore.ops.Lstsq`.
- [STABLE] Add operator primitive for `mindspore.ops.MatrixDiagPartV3`.
- [STABLE] Add operator primitive for `mindspore.ops.MatrixDiagV3`.
- [STABLE] Add operator primitive for `mindspore.ops.MatrixExp`.
- [STABLE] Add operator primitive for `mindspore.ops.MatrixPower`.
- [STABLE] Add operator primitive for `mindspore.ops.MaxPool3DWithArgmax`.
- [STABLE] Add operator primitive for `mindspore.ops.MaxUnpool2D`.
- [STABLE] Add operator primitive for `mindspore.ops.MultilabelMarginLoss`.
- [STABLE] Add operator primitive for `mindspore.ops.NextAfter`.
- [STABLE] Add operator primitive for `mindspore.ops.Orgqr`.
- [STABLE] Add operator primitive for `mindspore.ops.ReduceStd`.
- [STABLE] Add operator primitive for `mindspore.ops.RGBToHSV`.
- [STABLE] Add operator primitive for `mindspore.ops.RightShift`.
- [STABLE] Add operator primitive for `mindspore.ops.SampleDistortedBoundingBoxV2`.
- [STABLE] Add operator primitive for `mindspore.ops.ScaleAndTranslate`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterAddWithAxis`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterNdDiv`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterNdMax`.
- [STABLE] Add operator primitive for `mindspore.ops.ScatterNdMul`.
- [STABLE] Add operator primitive for `mindspore.ops.STFT`.
- [STABLE] Add operator primitive for `mindspore.ops.Trace`.
- [STABLE] Add operator primitive for `mindspore.ops.UpsampleNearest3D`.
- [STABLE] Add operator primitive for `mindspore.ops.UpsampleTrilinear3D`.
- [STABLE] Add distributed weight conversion interface `mindspore.parallel.transform_checkpoints`.
- [STABLE] Add distributed weight conversion interface `mindspore.parallel.transform_checkpoint_by_rank`.

#### Backwards Incompatible Change

##### Python API

- The `mindspore.ms_function` interface is renamed to `mindspore.jit`, and `mindspore.ms_function` will be deprecated and removed in a future version.
- The `mindspore.ms_class` interface is renamed to `mindspore.jit_class`, and `mindspore.ms_class` will be deprecated and removed in a future version.
- The `mindspore.ops.ms_kernel` interface is renamed to `mindspore.ops.kernel`, and `mindspore.ops.ms_kernel` will be deprecated and removed in a future version.
- The `mindspore.dataset.map` interface parameter `column_order` does not take effect, use`mindspore.dataset.project`.
- The `mindspore.dataset.close_pool` and `mindspore.dataset.to_device` and `mindspore.dataset.set_dynamic_columns` are deprecated and removed in this version.

### Bug fixes

- Fixed an issue where the mixed precision functional interface could not modify the backend driver in graph mode
- Fixed the problem that users can automatically transfer device_id in the single-P scenario for the following networks:（mobilenetv1/fasterrcnn/yolov3/yolov4/yolov5/unet/openpose/simplepose/crnn/gnmtv2/faceattribute/facequality/facedetection）

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

Contributions of any kind are welcome!

## MindSpore 1.10.1 Release Notes

### Bug fixes

- Fixed the issue that the specified axis is not considered in logsumexp anti-overflow processing
- Fixed the compilation dependency of proto file
- Fixed the issue that the print operator printing result is not normal
- Fixed the issue that the equal operator is out of range
- Fixed the problem that when function wrapped by @jit，the cell id is not correct
- Fixed the GNN scenario data type verification error
- Fixed the problem that the dataset.map multi-process degenerates into threads

### Contributors

Thanks goes to these wonderful people:

archer2049, caifubi, chenfei_mindspore, gaoshuanglong, Greatpan, guozhijian, huoxinyou, Kxiong, lanzhineng, lijunbin, liubuyu, liuchuting, luochao60, lyqlola, nomindcarry, TuDouNi, xiaotianci, xupan, yangshuo, yefeng, YingtongHu, yuchaojie, zhoufeng, ZPaC, 刘勇琪, 吕昱峰, 王禹程, 于振华.

Contributions of any kind are welcome!

## MindSpore 1.10.0 Release Notes

### Major Features and Improvements

#### DataSet

- [STABLE]The timeout waiting time is adjusted in data sinking mode. The default value is 600s after adjusted. This solves the isuses that the GetNext operator may timeout due to environment resource competition and large computing workload when training in sink mode.

### Bug fixes

- Fixed an issue where some Primitive operators in AMP cannot be instantiated in graph mode and the interface is unavailable.
- Fixed an issue of DynamicRNN execution failure in LSTM network under the scenario of computational force segmentation on Ascend platform.
- Fixed DEVICE_ID cannot be set by single card train scripts parameters in mobilenet, fasterrcnn, yolo, etc.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

Contributions of any kind are welcome!

## MindSpore Lite 1.10.0 Release Notes

### Bug fixes

- Fixed potential accuracy problem of arithmetic type CPU kernels at dynamical shape case.
- Fixed the Incorrect Write Address of the Deconv Quantization Operator.

## MindSpore 1.9.0 Release Notes

### Major Features and Improvements

#### FrontEnd

- [STABLE] Add the object-oriented and functional combination programming paradigm, add mixed-precision APIs for combination programming paradigms such as `mindspore.amp.LossScaler`, `mindspore.amp.DynamicLossScaler`, `mindspore.amp.StaticLossScaler`, `mindspore.amp.auto_mixed_precision` and `mindspore.amp.all_finite`.

### API Change

#### operator

- [STABLE] Add nn interface for `nn.AdaptiveAvgPool3d`.
- [STABLE] Add functional interface for `ops.adaptive_avg_pool3d`.
- [STABLE] Add functional interface for `ops.addcdiv`.
- [STABLE] Add functional interface for `ops.addcmul`.
- [STABLE] Add GPU and CPU support for `ops.approximate_equal`.
- [STABLE] Add GPU support for `ops.atanh`.
- [STABLE] Add GPU support for `ops.bessel_i0`.
- [STABLE] Add Ascend support for `ops.bessel_i0e`.
- [STABLE] Add GPU support for `ops.bessel_i1`.
- [STABLE] Add Ascend and GPU support for `ops.bessel_i1e`.
- [STABLE] Add GPU support for `ops.bessel_j0`.
- [STABLE] Add GPU support for `ops.bessel_j1`.
- [STABLE] Add GPU support for `ops.bessel_k0`.
- [STABLE] Add GPU support for `ops.bessel_k0e`.
- [STABLE] Add GPU support for `ops.bessel_k1`.
- [STABLE] Add GPU support for `ops.bessel_k1e`.
- [STABLE] Add GPU support for `ops.bessel_y0`.
- [STABLE] Add GPU support for `ops.bessel_y1`.
- [STABLE] Add functional interface for `ops.bias_add`.
- [STABLE] Add GPU support for `ops.bitwise_and`.
- [STABLE] Add GPU support for `ops.bitwise_or`.
- [STABLE] Add GPU support for `ops.bitwise_xor`.
- [STABLE] Add Ascend support for `ops.grid_sample`.
- [STABLE] Add CPU support for `ops.inplace_update`.
- [STABLE] Add Ascend and GPU support for `ops.isclose`.
- [STABLE] Add Ascend support for `ops.isnan`.
- [STABLE] Add GPU support for `ops.lerp`.
- [STABLE] Add functional interface for `ops.random_poisson`.
- [STABLE] Add functional interface for `ops.reverse_sequence`.
- [STABLE] Add GPU support for `ops.scatter_mul`.
- [STABLE] Add functional interface for `ops.scatter_nd_max`.
- [STABLE] Add functional interface for `ops.scatter_nd_min`.
- [STABLE] Add GPU support for `ops.SparseToDense`.
- [STABLE] Add functional interface for `ops.square`.
- [STABLE] Add GPU support for `ops.standard_laplace`.
- [STABLE] Add functional interface for `ops.std`.
- [STABLE] Add Ascend and GPU support for `ops.trunc`.
- [STABLE] Add functional interface for `ops.unsorted_segment_sum`.
- [STABLE] Add functional interface for `ops.xdivy`.
- [STABLE] Add GPU support for `ops.xlogy`.
- Deprecate `ops.poisson` and use `ops.random_poisson` instead.
- Deprecate `ops.SparseApplyAdagrad` and use `ops.SparseApplyAdagradV2` instead.

### Bug fixes

- [BUGFIX] The logic of the auto mixed precision (amp) O2 level is revised. In addition to the `BatchNorm1d` and `BatchNorm2d` operators, the other two operators `BatchNorm3d` and `LayerNorm` are added. The four operators still use the float32 data type when calculating.

- [BUGFIX] Fix the problem that when processing string type data, if `output_numpy=True` is specified when calling the `create_dict_iterator` or `create_tuple_iterator` interface, the obtained data will be of type `numpy.bytes_`. After this fixing, these interfaces will directly return `numpy.str_` type data, and users do not need to perform string decoding operations on it. Likewise, when performing user defined processing functions, the received data will also be of type `numpy.str_` directly, matching the original source data type.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, liyanliu, lizhenyu, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, panfengfeng, panyifeng, Payne, peixu_ren, Pengyongrong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanyuan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

Contributions of any kind are welcome!

## MindSpore 1.8.1 Release Notes

### API Change

#### operator

- [STABLE] Add GPU and CPU support for ops.ApplyAdagradDA.
- [STABLE] Add CPU support for ops.ApplyAdagradV2.
- [STABLE] Add Ascend dynamic shape support for ops.ApplyCenteredRmsProp.
- [STABLE] Add CPU support for ops.ApplyFtrl.
- [STABLE] Add CPU support for ops.ApplyGradientDescent.
- [STABLE] Add CPU support for ops.ApplyPowerSign.
- [STABLE] Add GPU and CPU support for ops.ApplyProximalAdagrad.
- [STABLE] Add Ascend dynamic shape support for ops.ApplyRmsProp.
- [STABLE] Add functional interface for ops.max.
- [STABLE] Add functional interface for ops.atan2.
- [STABLE] Add GPU support for ops.cummax.
- [STABLE] Add GPU and CPU support for ops.cummin.
- [STABLE] Add GPU support for ops.diag.
- [STABLE] Add functional interface for ops.expand_dims.
- [STABLE] Add functional interface for ops.gather_elements.
- [STABLE] Add GPU support for ops.grid_sample.
- [STABLE] Add Ascend support for ops.hardswish.
- [BETA] Add GPU support for ops.index_fill.
- [BETA] Add CPU support for ops.inplace_update.
- [BETA] Add GPU support for nn.InstanceNorm1d.
- [BETA] Add GPU support for nn.InstanceNorm2d.
- [BETA] Add GPU support for nn.InstanceNorm3d.
- [STABLE] Add functional interface for ops.log1p.
- [STABLE] Add GPU and CPU support for ops.masked_fill.
- [BETA] Add GPU support for ops.matrix_diag_part.
- [BETA] Add GPU support for ops.matrix_diag.
- [BETA] Add GPU support for ops.matrix_set_diag.
- [STABLE] Add GPU support for ops.max_pool3d.
- [STABLE] Add functional interface for ops.nll_loss.
- [STABLE] Add functional interface for ops.one_hot.
- [STABLE] Add functional interface for ops.pad.
- [STABLE] Add CPU support for ops.random_gamma.
- [STABLE] Add functional interface for ops.amax.
- [STABLE] Add functional interface for ops.mean.
- [STABLE] Add functional interface for ops.amin.
- [STABLE] Add functional interface for ops.prod.
- [STABLE] Add Ascend, GPU, and CPU support for ops.renorm.
- [BETA] Add Ascend, GPU, and CPU support for ops.tensor_scatter_elements.
- [STABLE] Add GPU support for ops.scatter_max.
- [STABLE] Add GPU support for ops.scatter_min.
- [STABLE] Add functional interface for ops.scatter_nd.
- [STABLE] Add GPU support for ops.scatter_nd_max.
- [STABLE] Add functional interface for ops.scatter_update.
- [STABLE] Add CPU support for ops.binary_cross_entropy_with_logits.
- [STABLE] Add functional interface for ops.smooth_l1_loss.
- [STABLE] Add CPU support for ops.space_to_batch_nd.
- [STABLE] Add GPU and CPU support for ops.SparseApplyAdagrad.
- [STABLE] Add GPU and CPU support for ops.sparse_segment_mean.
- [STABLE] Add functional interface for ops.squeeze.
- [STABLE] Add CPU support for ops.standard_laplace.
- [BETA] Add Ascend, GPU, and CPU support for nn.ReflectionPad1d.
- [BETA] Add Ascend, GPU, and CPU support for nn.ReflectionPad2d.
- [STABLE] Add Ascend, GPU, and CPU support for nn.SiLU.
- [STABLE] Add functional interface for ops.transpose.
- [STABLE] Add CPU support for ops.uniform_candidate_sampler.
- [STABLE] Add functional interface for ops.uniform.
- [STABLE] Add GPU support for ops.unique_with_pad.
- [STABLE] Add functional interface for ops.unstack.
- [BETA] Add GPU and CPU support for ops.interpolate.
- [STABLE] Add CPU support for ops.xdivy.
- [STABLE] Add CPU support for ops.xlogy.

## MindSpore 1.8.0 Release Notes

### Major Features and Improvements

#### FrontEnd

- [BETA] Add `mindspore.train.Model.fit` API, add `mindspore.train.callback.EarlyStopping` and `mindspore.train.callback.ReduceLROnPlateau` in Callback.
- [BETA] Support custom operator implemented by Julia.
- [BETA] Support custom operator implemented by MindSpore Hybrid DSL.
- [STABLE] The export() interface supports the export of a model using a custom encryption algorithm, and the load() interface supports the import of a model using a custom decryption algorithm.
- [BETA] [Unified_Dynamic_and_Static_Graphs] [Usability] Constant-type data (tuple/list/dict is supported in Version 1.8) can be set to be variable during graph compiling.
- [BETA] [Unified_Dynamic_and_Static_Graphs] JIT fallback is used to support the control flow capability in the constant scenario.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The Python raise statement is supported in the graph mode constant scenario.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The Python assert statement is supported in the graph mode constant scenario.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The Python print statement is supported in the graph mode constant scenario.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The str.format() method is supported in the graph mode.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The slice method can be used to assign a value to the list in the graph mode.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] The instances of custom classes can be created and invoked in the graph mode.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] Obtaining the properties of a class from the Cell array and the custom class array is supported.
- [STABLE] [Unified_Dynamic_and_Static_Graphs] isinstance supports scenario expanding in the graph mode.
- [STABLE] Rename the custom operator decorator 'ms_hybrid' to 'ms_kernel'.
- [BETA] Custom operator Hybrid DSL is supported on the backend of CPU.
- [BETA] Custom operator Ascend backend adds custom scheduling primitive syntax support.

#### PyNative

- [STABLE] Implement the AdamWeightDecay operator to replace the original small operator combination mode.
- [STABLE] In PyNative mode, execute the optimizer by unifying the dynamic and static graphs.
- [STABLE] Optimize the execution performance of PyNative bprop graph and ms_function.

#### Auto Parallel

- [STABLE] Docking the AllToAll single-operator mode. Support AllToAll Operator in the graph compilation level O0.
- [STABLE] Whole-graph offloading supports MPI launching. In Whole-graph offloading, launching with MPI is supported.
- [STABLE] Seeds of model weights provide parallel interface configuration. If you do not set the random number of seeds through the mindspore.set_seed command, the weights initialized by each parameter is determined by the current fragment index. If the random number of seeds are configured, the initialization results of the same shape and weight of the same segmentation policy are the same.
- [STABLE] The HCCL shields internal full-mesh and non-full-mesh connections. Both fully-connected AllToAllv and hierarchical AllToAllv are allowed in one training session.
- [BETA] CPU optimizer fusion. Multiple optimizer operators are combined according to data types through cross-parameter fusion, improving performance. Currently, It has been verified on CPU AdamWeightDecay optimizer. You can use the flatten_weights method in the network cell class to enable this function.

#### Executor

- [STABLE] Provide southbound API.
- [STABLE] Multi-actor fusion execution to optimize the execution performance during runtime.
- [STABLE] Nopop operators (eg. reshape) execute elimination.
- [STABLE] Embedded cache architecture switches unified distributed runtime.
- [STABLE] Parameter Server switches unified distributed runtime.
- [STABLE] Support Parameter Server mode training on CPU.

#### DataSet

- [STABLE] When using the map operation for dataset objects and the parameters like: num_parallel_workers > 1 and python_multiprocessing=True, the multi-process mechanism is optimized, so that the data channel and child processes are mapped one by one, avoiding excessive file handle occupation, and closing_pool interface is also deleted.
- [STABLE] Add a batch of Vision, Text and Audio data augmentation operations.
- [STABLE] Fix a bug where the flat_map method of the Dataset class does not flatten the result.
- [STABLE] Unify import paths of dataset augmentation APIs to provide more easier way to use. Refer to [latest api usages](https://www.mindspore.cn/docs/en/r1.8/api_python/mindspore.dataset.vision.html).

### API Change

#### operator

- [STABLE] Add GPU support for ops.adaptive_avg_pool2d.
- [BETA] Add Ascend, GPU, and CPU support for ops.adaptive_max_pool2d .
- [BETA] Add CPU support for ops.approximate_equal.
- [STABLE] Add CPU support for ops.argmin.
- [BETA] Add CPU support for ops.assign_sub.
- [STABLE] Add GPU support for ops.bernoulli.
- [BETA] Add CPU support for ops.bessel_i0.
- [BETA] Add CPU support for ops.bessel_i0e.
- [BETA] Add CPU support for ops.bessel_i1.
- [BETA] Add CPU support for ops.bessel_i1e Add CPU support.
- [STABLE] Add CPU support for ops.bessel_j0.
- [STABLE] Add CPU support for ops.bessel_j1.
- [STABLE] Add CPU support for ops.bessel_k0.
- [STABLE] Add CPU support for ops.bessel_k0e.
- [BETA] Add CPU support for ops.bessel_k1.
- [BETA] Add CPU support for ops.bessel_k1e.
- [STABLE] Add CPU support for ops.bessel_y0.
- [STABLE] Add CPU support for ops.bessel_y1.
- [STABLE] Add CPU support for ops.bitwise_and.
- [STABLE] Add CPU support for ops.bitwise_or.
- [STABLE] Add CPU support for ops.bitwise_xor.
- [STABLE] Add functional interface for ops.broadcast_to.
- [BETA] Add GPU and CPU support for ops.ceil.
- [BETA] Add GPU support for ops.col2im.
- [BETA] Add functional interface for ops.concat.
- [STABLE] Add GPU support for ops.cosh.
- [STABLE] Add Ascend and CPU support for ops.ctc_greedy_decoder.
- [BETA] Add GPU and CPU support for ops.DataFormatDimMap.
- [BETA] Add GPU and CPU support for ops.dropout2d.
- [BETA] Add CPU support for ops.dropout3d.
- [BETA] Add CPU support for ops.erf.
- [BETA] Add CPU support for ops.erfc.
- [STABLE] Add functional interface for ops.expand_dims.
- [STABLE] Add GPU and CPU support for ops.fast_gelu.
- [STABLE] Add Ascend dynamic shape support for ops.flatten.
- [BETA] Add GPU and CPU support for ops.ger.
- [STABLE] Add Ascend, GPU, and CPU support for ops.gumbel_softmax.
- [BETA] Add GPU and CPU support for ops.hardshrink.
- [BETA] Add CPU support for ops.index_add.
- [BETA] Add CPU support for ops.inplace_add.
- [BETA] Add CPU support for ops.inplace_sub.
- [STABLE] Add CPU support for ops.intopk.
- [STABLE] Add GPU and CPU support for ops.inv.
- [STABLE] Add GPU and CPU support for ops.invert.
- [BETA] Add CPU support for ops.isclose.
- [STABLE] Add CPU support for ops.lerp.
- [BETA] Add CPU support for ops.linspace.
- [BETA] Add functional interface for ops.log_softmax.
- [BETA] Add Ascend, GPU, and CPU support for ops.norm.
- [BETA] Add CPU support for ops.lrn.
- [BETA] Add GPU support for ops.masked_select.
- [BETA] Add GPU and CPU support for ops.matrix_band_part.
- [BETA] Add GPU and CPU support for ops.matrix_solve.
- [BETA] Add CPU support for ops.meshgrid.
- [STABLE] Add CPU support for ops.mish.
- [BETA] Add GPU support forops.nonzero.
- [STABLE] Add GPU and CPU support for ops.padding.
- [BETA] Add Ascend dynamic shape support for ops.pow.
- [BETA] Add functional interface for ops.range.
- [BETA] Add Ascend dynamic shape support for ops.round.
- [STABLE] Add Ascend dynamic shape support for ops.scatter_add.
- [STABLE] Add Ascend dynamic shape support for ops.scatter_div.
- [BETA] Add GPU support for ops.scatter_max.
- [BETA] Add GPU support for ops.scatter_min.
- [BETA] Add CPU support for ops.scatter_nd_add.
- [STABLE] Add GPU and CPU support for ops.scatter_nd_div.
- [STABLE] Add GPU and CPU support for ops.scatter_nd_min.
- [STABLE] Add GPU and CPU support for ops.scatter_nd_mul.
- [BETA] Add CPU support for ops.scatter_nd_sub.
- [STABLE] Add Ascend dynamic shape support for ops.scatter_update.
- [BETA] Add Ascend dynamic shape support for ops.select.
- [BETA] Add GPU and CPU support for ops.selu.
- [BETA] Add GPU and CPU support for ops.soft_shrink.
- [BETA] Add CPU support for ops.softsign.
- [STABLE] Add GPU support for ops.tan.
- [BETA] Add Ascend and CPU support ops.tensor_scatter_add.
- [STABLE] Add GPU and CPU support for ops.tensor_scatter_div.
- [STABLE] Add GPU and CPU support for ops.tensor_scatter_mul.
- [BETA] Add Ascend and CPU support for ops.tensor_scatter_sub.
- [STABLE] Add Ascend, GPU, and CPU support for nn.AdaptiveAvgPool1d.
- [STABLE] Add Ascend, GPU, and CPU support for nn.AdaptiveMaxPool1d.
- [BETA] Add Ascend, GPU, and CPU support for nn.BiDense.
- [STABLE] Add Ascend, GPU, and CPU support for nn.ConstantPad1d.
- [STABLE] Add Ascend, GPU, and CPU support for nn.ConstantPad2d.
- [STABLE] Add Ascend, GPU, and CPU support for nn.ConstantPad3d.
- [STABLE] Add Ascend, GPU, and CPU support for nn.Hardtanh.
- [STABLE] Add Ascend, GPU, and CPU support for nn.HuberLoss.
- [STABLE] Add Ascend, GPU, and CPU support for nn.RReLU.
- [STABLE] Add Ascend, GPU, and CPU support for nn.Tanhshrink.
- [STABLE] Add Ascend, GPU, and CPU support for nn.Threshold.
- [STABLE] Add Ascend, GPU, and CPU support for nn.ZeroPad2d.
- [BETA] Add GPU support for ops.unique_consecutive.
- [STABLE] Add CPU support for ops.unsorted_segment_max.
- [STABLE] Add CPU support for ops.unsorted_segment_min.
- [STABLE] Add GPU support for ops.unsorted_segment_prod.

#### Backwards Incompatible Change

##### Python API

- DVPP simulation algorithm is no longer supported. Remove `mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg` and `mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg` interfaces.
- Add `on_train_epoch_end` method in LossMonitor, which implements printing metric information in the epoch level when it is used in `mindspore.train.Model.fit`.
- TimeMonitor printing content changes, and the printed content is added to "train" or "eval" to distinguish between training and inference phases.
- `filter_prefix` of `mindspore.load_checkpoint` interface: empty string ("") is no longer supported, and the matching rules are changed from strong matching to fuzzy matching.

#### Import Optimization

APIs in `mindspore.context`, `mindspore.parallel`, `mindspore.profiler` and `mindspore.train` can be directly used in `mindspore`. The original usage can still be supported.

For examples:

- `mindspore.context.set_context` can be simplified to `mindspore.set_context`.
- `mindspore.parallel.set_algo_parameters` can be simplified to `mindspore.set_algo_parameters`.
- `mindspore.profiler.Profiler` can be simplified to `mindspore.Profiler`.
- `mindspore.train.callback.Callback` can be simplified to `mindspore.train.Callback`.

The API pages are aggregated to <https://www.mindspore.cn/docs/en/r1.8/api_python/mindspore.html>.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

Contributions of any kind are welcome!

## MindSpore Lite 1.8.0 Release Notes

### Major Features and Improvements

#### API

- [STABLE] Add C++ and Python APIs for model conversion.
- [STABLE] Add Python APIs for model inference.

#### Post-Training Quantization

- [STABLE] Support perlayer quantization, and built-in CLE to optimize perlayer quantization accuracy.

## MindSpore 1.7.0 Release Notes

### Major Features and Improvements

#### OS

- [STABLE] Support Python 3.8 (Linux/Windows/Mac).
- [STABLE] Installation improved with more detailed install guide and automated shell scripts.
- [STABLE] Support operator computing with multi-thread under Windows.
- [STABLE] Compatible with GCC from version 7.3 to 9.x.

#### FrontEnd

- [STABLE] Support dynamic weight decay for optimizers, that is weight decay value will change according to the increasing step during training.
- [STABLE] Add four methods to create Tensor, which are `mindspore.numpy.rand()`, `mindspore.numpy.randn()`, `mindspore.numpy.randint()`, and `mindspore.ops.arange()`.
- [STABLE] Add `mindspore.train.callback.History` in Callback.
- [BETA] Support custom operator implemented by Julia operator.
- [STABLE] Support accessing attributes and methods of user-defined classes  through `mindspore.ms_class` class decorator.
- [STABLE] Support training when a network has side effect operations and control flow statements at the same time.
- [STABLE] Support for more complex control flow syntax, such as a for loop statement in the body of a while loop.
- [STABLE] Improve the performance of networks with complex syntax control flow statements by decreasing the num of subgraphs.

#### PyNative

- [STABLE] Add Hook functions in PyNative mode, including register_forward_pre_hook, register_forward_hook of the forward hook interface, register_backward_hook of the reverse hook interface.
- [STABLE] Optimize the execution performance of PyNative mode, and execute the front-end Python and the back-end C++ in parallel.

#### Auto Parallel

- [STABLE] Support TopK routing, data parallel and optimizer state parallel when enable MoE.
- [STABLE] Support AllGather/ReduceScatter communication operator fusion. Support AllReuduce fusion by the data volume size in DATA_PARALLEL mode.
- [STABLE] Support ops.clip_by_global_norm in the parallel mode.
- [STABLE] Support AdaSum optimizer in the parallel mode.
- [STABLE] Support automatic optimizer state parallel.
- [STABLE] Support AlltoAll configurable. Support automatically add virtualdataset cell.
- [STABLE] Support automatically inference trainable parameters in pipeline parallel training.
- [STABLE] Support clusters where the device number is not the power of 2.
- [STABLE] Support sharding propagation in auto-parallel mode.
- [STABLE] Support optimizer offload under the unified runtime.
- [STABLE] Support Adafactor operator on CPU.
- [STABLE] Support sharding at H/W axis for Conv2d/Conv2DTranspose operator. Support operators such as ResizeBilinear，ROIAlign, CropAndResize, BoundingBoxEncode, IOU and RandomChoiceWithMask.

#### Executor

- [BETA] [Failure Recovery Under Data Parallel Training](https://www.mindspore.cn/tutorials/experts/en/r1.7/parallel/train_gpu.html) Support auto failure recovery under data parallel training mode.
- [BETA] Support searching for the number of threads under the CPU to obtain the optimal number of threads for execution. The entire search process takes 50 steps, and the overall performance will reach a stable state after 50 steps. When testing performance, data after 50 steps need to be used as a standard.

#### DataSet

- [STABLE] Add dataset operations mapping between TensorFlow.data module and MindSpore.dataset module, [check list](https://www.mindspore.cn/docs/en/r1.7/note/api_mapping/tensorflow_api_mapping.html#tf-data).
- [STABLE] Python multiprocessing optimization and make processes exit normally.
- [STABLE] Support [Dataset Autotune](https://www.mindspore.cn/tutorials/experts/en/r1.7/debug/dataset_autotune.html) for tuning the speed of dataset pipeline automatically.
- [BETA] [Dataset Offload](https://www.mindspore.cn/docs/en/r1.7/design/dataset_offload.html) support new data augmentation operations: RandomColorAdjust, RandomSharpness, TypeCast.
- Output a single data column when `__getitem__/__next__` methods of GeneratorDataset return a single NumPy object.
- Use `ulimit -u 10240` to increase the number of threads/processes available to the current user when specify too many processes or threads for loading dataset may cause RuntimeError: can't start new thread.

### API Change

#### Backwards Incompatible Change

##### Python API

- Modify the gradient return value type of the hook corresponding to the register_backward_hook function, and change the gradient return value to the tuple type uniformly.([!31876](https://gitee.com/mindspore/mindspore/pulls/31876))
- Deprecated usage: `import mindspore.dataset.engine.datasets as ds`. Use `import mindspore.dataset as ds` instead as recommended in [mindspore doc](https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore.dataset.html).
- Add `mindspore.ms_class` interface, as class decorator for user-defined classes. It allows MindSpore to identify user-defined classes and access their attributes and methods([!30855](https://gitee.com/mindspore/mindspore/pulls/30855))
- Deprecate `mindspore.SparseTensor` and use `mindspore.COOTensor` instead. ([!28505](https://gitee.com/mindspore/mindspore/pulls/28505))
- Add Tensor init arg `internal` for internal use.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

## MindSpore Lite 1.7.0 Release Notes

### Major Features and Improvements

#### Post quantization

- [STABLE] Support post quantization to run dynamic quantization algorithm.
- [BETA] Support post quantized model to run on NVIDIA GPU.

# MindSpore 1.6.0

## MindSpore 1.6.0 Release Notes

### Major Features and Improvements

#### OS

- [STABLE] Support macOS with CPU(X86)
- [BETA] Supoport macOS with CPU(M1)

#### FrontEnd

- [STABLE] Support JIT Fallback feature in Graph mode.
- [STABLE] Support compile cache feature in Graph mode.
- [STABLE] Add new optimizers, including ASGD and Rprop.
- [STABLE] Add new initializers, including Identity, Orthogonal, Dirac, Sparse and VarianceScaling.
- [STABLE] Support resuming training when an exception occurs in the process.
- [STABLE] Change `mindspore.nn.LSTMCell` from single-layer LSTM to single-cell LSTM.
- [BETA] Introduce `mindspore.ops.Custom` to customize your own operators for Ascend(AICore, AICPU), GPU, CPU backends, and the custom type can be one of TBE, AKG, pure Python function or prebuild binary(called aot operator).

#### PyNative

- [STABLE] Support heterogeneous feature in PyNative mode.
- [STABLE] Optimize memory allocation in PyNative mode.

#### Auto Parallel

- [STABLE] Support configuring the output shard strategy of the MatMul distributed operator.
- [STABLE] Support multi-instances parallel.
- [STABLE] Support activation slice communication and calculation overlap in Transformer.
- [STABLE] Support heterogeneous parallel tensor swap.
- [STABLE] Add implementations of distributed operator of ResizeNearestNeighbor.
- [STABLE] Add a communication operator named NeighborExchangeV2 that supports data exchange between adjacent 8 rank ids.
- [STABLE] Pipeline parallel support GPU platform.
- [STABLE] Add cell-level data parallel interface.
- [STABLE] Support gradient AllReduce fusion according to the amount of data.
- [STABLE] Support a sharding strategy search algorithm called sharding propagation.

#### Executor

- [STABLE] Support multigraph sink and subgraph sink of MindRT.
- [STABLE] Support memory swap to break the device memory size limit on Ascend platform.
- [STABLE] Support dynamic deployment of distributed training cluster(GPU).
- [BETA] Support automatic failover of parameter server.

#### DataSet

- [STABLE] Support overwrite feature in MindRecord.
- [STABLE] Log improvement and more friendly to users.
- [BETA] Support new feature [Dataset Offload](https://www.mindspore.cn/docs/programming_guide/en/r1.6/enable_dataset_offload.html) to speed up data processing by heterogeneous computing.
- [BETA] Support new feature [Dataset Autotune](https://www.mindspore.cn/docs/programming_guide/en/r1.6/enable_auto_tune.html) to adjust parallelism of dataset pipeline automatically.

#### GraphKernel Fusion

- [STABLE] Support kernel fusion and generation for CPU backend.

#### Federated Learning

- [STABLE] FL-Client framework and model decoupling.
- [BETA] Support Cross-silo federated learning framework.

#### Debug

- [STABLE] Support dump in cell level(Ascend).
- [STABLE] Support dump Tensor statistics(Ascend/GPU).
- [STABLE] Support displaying corresponding code lines for fusion nodes.
- [STABLE] Support passing dump flag in Ascend backend in order to dump correct operators after fusion transformation.

### API Change

#### Backwards Incompatible Change

##### Python API

###### `mindspore.dataset.MindDataset` interface changes input parameter dataset_file([!27542](https://gitee.com/mindspore/mindspore/pulls/27542))

`MindDataset` contains the input parameter `dataset_file`, which is in the singular format. It can receive a single file path or a list that stores multiple file paths. Thus It is preferred to change the input parameter `dataset_file` into plural format. In addition, the input parameters of most dataset API, such as `TFRecordDataset`, are in plural formart (`dataset_files`). To ensure consistency, the input parameter `dataset_file` of MindDataset is changed to plural formart as `dataset_files`,  we can see the updated version in api of [mindspore.dataset.MindDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset).

###### Delete `mindspore.Tensor`'s property `virtual_flag`([!26989](https://gitee.com/mindspore/mindspore/pulls/26989))

###### Delete `mindspore.Parameter`'s property `is_init`([!26989](https://gitee.com/mindspore/mindspore/pulls/26989))

###### Delete `mindspore.nn.ROC`'s interface `roc`([!25713](https://gitee.com/mindspore/mindspore/pulls/25713))

###### The `shard()` interface of primitive is changed from `shard(strategy)` to `shard(in_strategy=None, out_strategy=None)`

###### The `set_auto_parallel_context()` interface of context is changed from

###### `set_auto_parallel_context(parallel_mode=AUTO_PARALLEL, auto_parallel_search_mode="dynamic_programming")` to `set_auto_parallel_context(parallel_mode=AUTO_PARALLEL, search_mode="dynamic_programming")`

#### Collect Data and Create Landscape

##### Python API

###### `mindspore.train.callback.SummaryCollector` interface's parameter `collect_specified_data` add new operations `collect_landscape` ([!26229](https://gitee.com/mindspore/mindspore/pulls/26229))

`collect_landscape` can collect the parameters needed to create the loss landscape. we can see the updated version in api of [mindspore.train.callback.SummaryCollector](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryCollector.html#mindspore.SummaryCollector).

###### `mindspore.train.callback` add new interface `SummaryLandscape` ([!26229](https://gitee.com/mindspore/mindspore/pulls/26229))

`SummaryLandscape` can help you to collect loss landscape information. It can create landscape in PCA direction or random direction by calculating loss. We can see the updated version in api of [mindspore.train.callback.SummaryLandscape](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryLandscape.html#mindspore.SummaryLandscape).

### Bug fixes

#### Executor

- Fix process hanging while calling MPI_comm_create in asymmetric pipeline split scenario. ([!28707](https://gitee.com/mindspore/mindspore/pulls/28707))
- Fix the execution error when the weights are shared between graph mode and PyNative mode.([!26635](https://gitee.com/mindspore/mindspore/pulls/26635))
- Fixed the probability coredump when free memory under PyNative mode.([!25472](https://gitee.com/mindspore/mindspore/pulls/25472))

#### Dataset

- Fix memory increase abnormally when running dataset for a long time. ([!26237](https://gitee.com/mindspore/mindspore/pulls/26237))
- Fix saving MindRecord files with Chinese path on Windows. ([!28378](https://gitee.com/mindspore/mindspore/pulls/28378))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

- [STABLE] Add more fusion patterns in the converter tool to improve runtime performance.
- [STABLE] Support take OpenGL texture as input and output of inference.
- [STABLE] Refactor the JAVA API.
- [BETA] Support inference on Ascend310.

#### x86 backend optimization

- [STABLE] Optimize kernels for x86 using Advanced Vector Extensions(AVX512).

#### ARM backend optimization

- [STABLE] Support heterogeneous parallel inference, including splitting operators, constructing heterogeneous subgraphs, and heterogeneous parallel scheduling between CPUs and GPUs.
- [STABLE] Add more FP16 operators.

#### Post quantization

- [STABLE] Post quantization supports debugging.
- [STABLE] Full quantization supports choosing non-quantized nodes.
- [STABLE] Mixed bit quantization supports auto-tune.

#### Training on Device

- [STABLE] Support user-defined algorithm models to access the federated learning framework.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, [wangnan39@huawei.com](mailto:wangnan39@huawei.com), wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, [zhanghaibo5@huawei.com](mailto:zhanghaibo5@huawei.com), zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.5.2

## MindSpore 1.5.2 Release Notes

### Bug fixes

- Fix code specification, pclint, codedex alarm.
- Repair NN Abnormal output of graphnorm operator.
- Fixed the problem of poor performance in scenes with dynamic rnngrad batch size of 16 times.

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.5.1

## MindSpore 1.5.1 Release Notes

### Bug fixes

- Fix code specification, pclint, codedex alarm.
- Fix yolov4 network probabilistic segment error.

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.5.0

## MindSpore 1.5.0 Release Notes

### Major Features and Improvements

#### NewModels

- [STABLE] Add CV model on Ascend: Fast-SCNN
- [BETA] Add CV models on Ascend: midas_V2, attgan, FairMOT, CenterNet_resnet101, SEResNext, YOLOV3-tiny, RetinaFace
- [STABLE] Add CV models on GPU: ssd_mobilenetv1_fpn, shufflenetv1, tinyDarkNet, CNN-CTC, unet++, DeepText, SqueezeNet
- [STABLE] Add NLP models on GPU: GRU, GNMT2, Bert-Squad
- [STABLE] Add recommend models on GPU: NCF
- [BETA] Add CV models on GPU: FaceAttribute, FaceDetection, FaceRecongnition SENet,
- [BETA] Add Audio models on GPU: DeepSpeech2
- [STABLE]`model_zoo` has been separated to an individual repository`models`

#### FrontEnd

- [STABLE] Support`while` and`break`,`continue` statements of training network in`GRAPH_MODE`.
- [BETA] Support export MindIR file after model training in cloud side and evaluate in edge side by import the MindIR file.
- [STABLE] Support forward mode auto-diff interface Jvp(Jacobian-Vector-Product).
- [STABLE] Support backward mode auto-diff interface Vjp(Vector-Jacobian-Product).

#### Auto Parallel

- [STABLE] Support distributed pipeline inference.
- [STABLE] Add implementation of the sparse attention and its distributed operator.
- [STABLE] Add implementations of distributed operator of Conv2d/Conv2dTranspose/Conv2dBackpropInput/Maxpool/Avgpool/Batchnorm/Gatherd.
- [STABLE] Support configuring the dataset strategy on distributed training and inference mode.
- [STABLE] Add high level API of the Transformer module.

#### Executor

- [STABLE] Support AlltoAll operator.
- [STABLE] CPU operator (Adam) performance optimization increased by 50%.
- [BETA] Support Adam offload feature, reduce the static memory usage of Pangu large model by 50%.
- [STABLE] MindSpore Ascend backend supports configuration operator generation and loading cache path.
- [STABLE] MindSpore Ascend backend supports lazy build in PyNaitve mode and compilation performance improved by 10 times.
- [STABLE] The function or Cell decorated by ms_function supports gradient calculation in PyNative mode.
- [STABLE] The outermost network supports parameters of non tensor type in PyNative mode.

#### DataSet

- [BETA] Add a new method for class Model to support auto data preprocessing in scenario of Ascend 310 inference.
- [STABLE] Add a new drawing tool to visualize detection/segmentation datasets.
- [STABLE] Support a new tensor operation named ConvertColor to support color space transform of images.
- [STABLE] Enhance the following tensor operations to handle multiple columns simultaneously: RandomCrop, RandomHorizontalFlip, RandomResize, RandomResizedCrop, RandomVerticalFlip.
- [STABLE] Support electromagnetic simulation dataset loading and data augmentation.
- [STABLE] Optimize the error logs of Dataset to make them more friendly to users.

#### Federated Learning

- [STABLE] Change the deployment environment of FL-Client.

#### Running Data Recorder

- [STABLE] RDR saves collected data files within directories named by Rank ID on distributed training on Ascend, GPU and CPU.

#### GraphKernel Fusion

### API Change

#### Backwards Incompatible Change

##### Python API

###### New Recomputation Configuration for AutoParallel and SemiAutoParallel Scenarios

Configuring the recomputation of the communication operations generated by the model parallel and optimizer parallel to save the memory on the
devices. Users can pass `mp_comm_recompute` and `parallel_optimizer_comm_recompute` to enable the recomputation of the communication operations.

### Bug fixes

#### FrontEnd

- Fix bug of too many subgraphs when network include`for` statement.([!23669](https://gitee.com/mindspore/mindspore/pulls/23669))

#### Executor

- RunTask failed when parameter_broadcast is enabled in PyNative mode. ([!23255](https://gitee.com/mindspore/mindspore/pulls/23255))
- An illegal memory access was encountered in the dynamic shape net on GPU.
- Fix tune failed for DynamicRnn. ([!21081](https://gitee.com/mindspore/mindspore/pulls/21081))

#### Dataset

- Optimize thread monitoring to solve the problem of running multiple multiprocessesing on Windwos. ([!23232](https://gitee.com/mindspore/mindspore/pulls/23232))
- Fix bugs of Dataset tensor operations in lite mode. ([!21999](https://gitee.com/mindspore/mindspore/pulls/21999))
- Fix memory increasing when using create_dict_iterator in for loop. ([!22529](https://gitee.com/mindspore/mindspore/pulls/22529))([!22529](https://gitee.com/mindspore/mindspore/pulls/22529))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Optimize TDNN-like streaming model by reusing the result of last inference.
2. Support dynamic filter Convolution.
3. Support serializing float32 weight into float16 weight for reducing size of model file.
4. Provide unified runtime API for developer reusing their code between cloud side and end side.
5. Now developer can configure built-in pass as custom passes.
6. Now user can specify format and shape of model inputs while converting model.
7. Support multiple devices inference, includeing CPU, NPU, GPU. User can set devices in mindspore::Context.
8. Support mixed precision inference. User can set inference precision by LoadConfig API.
9. Support custom operator registration and enable inference on third-party hardware.

#### ARM backend optimization

1. Support the nchw data format of some Operators, such as Conv, InstanceNorm, etc. The performance of some models convertered from onnx and caffe is greatly improved.
2. Fix bugs of memory leak on NPU.

#### Post quantization

1. Weight quantization supports mixed bit quantization.
2. Full quantization supports data pre-processing.
3. Adjust the quantization parameters from the command line to the configuration file.

#### Training on Device

1. Unify lite external api with MindSpore.
2. Implement static memory allocator and common workspace for TOD，save memory 10-20%.
3. Provide getgradients and setgradients interface，get and set optimizer params interfaces to support MOE Model.
4. Support user specified output node when export IOD Model.
5. Support more text  networks (tinybert,albert) and operators.

#### Codegen

1. Support kernel register for custom op. Third-party hardware like NNIE can be accessed through it.

### API Change

#### API Incompatible Change

##### C++ API

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.4.0

## MindSpore 1.4.0 Release Notes

### Major Features and Improvements

#### NewModels

#### FrontEnd

#### Auto Parallel

- Add distributed operators: Conv2D/Conv2DTranspose/Conv2DBackpropInput/MaxPool/AvgPool/BatchNorm/GatherD
- Support to configure shard strategy for dataset

#### Executor

#### DataSet

- Add SlicePatchesOperation for Remote Sensing feature（[!18179](https://e.gitee.com/mind_spore/repos/mindspore/mindspore/pulls/18179)）

#### FederatedLearning

#### Running Data Recorder

#### GraphKernel Fusion

#### Profiler

- [STABLE]  Support MS_DIAGNOSTIC_DATA_PATH for profiler feature.(Ascend/GPU)

#### Dump

- [STABLE]  Support MS_DIAGNOSTIC_DATA_PATH for dump feature.(Ascend/GPU/CPU)

### API Change

#### Backwards Incompatible Change

##### Python API

##### Command Line Interface

###### Dump Config

Previously, we need to set the dump path in dump config file. To make the dump feature easier to use on cloud, we support new environment parameter `MS_DIAGNOSTIC_DATA_PATH`.

| 1.3.0                          | 1.4.0                                                                                                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `path` is a mandatory field. | `path` field is optional.  If `path` field is not provided or is empty string, `MS_DIAGNOSTIC_DATA_PATH` should be set in environment. |

### Bug fixes

#### FrontEnd

#### Executor

#### Dataset

- Fix module 'signal' has no attribute 'SIGCHLD' problem under windows platform. ([!21232](https://gitee.com/mindspore/mindspore/pulls/21232))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

#### x86 backend optimization

#### ARM backend optimization

#### Cuda backend optimization

#### OpenCL backend

#### Post quantization

#### Training on Device

#### Codegen

### API Change

#### API Incompatible Change

##### C++ API

#### New features

##### Java API

### Bug fixes

#### Deprecations

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.3.0

## MindSpore 1.3.0 Release Notes

### Major Features and Improvements

#### NewModels

- [STABLE] Add CV models on Ascend: CPM, FCN8s, SSD-ResNet50-FPN, EAST, AdvancedEast.
- [STABLE] Add NLP models on Ascend: DGU, TextCNN, SentimentNet(LSTM).
- [STABLE] Add CV models on GPU: Faster-RCNN, FCN8s, CycleGAN, AdvancedEast.
- [BETA] Add CV models on Ascend: CycleGAN, PoseNet, SimCLR.
- [BETA] Add NLP models on Ascend: DGU, EmoTect, Senta, KT-Net.
- [BETA] Add NLP models on GPU: DGU, EmoTect.
- [BETA] Add EPP-MVSNet: a novel deep learning network for 3D reconstruction from multi-view stereo, which has won the first place in Tanks & Temples leaderboard(until April 1, 2021)(GPU).

#### FrontEnd

- [STABLE] The default running mode of MindSpore is changed to Graph mode.
- [STABLE] Support interface `run_check` to check whether MindSpore is working properly or not.
- [STABLE] Support saving custom information in the checkpoint file.
- [STABLE] Normal class adds mean parameter.
- [STABLE] Support export YOLOv3-DarkNet53 and YOLOv4 ONNX model.
- [STABLE] Support 40+ operator export ONNX model.
- [STABLE] The Metric module supports `set_indexes` to select the inputs of `update` in the specified order.
- [STABLE] Switch `_Loss` to an external API `LossBase` as the base class of losses.

#### Auto Parallel

- [STABLE] Add distributed operators: Select/GatherNd/ScatterUpdate/TopK.
- [STABLE] Support basic pipeline parallelism.
- [STABLE] Optimize sharding strategy setting of `Gather`.
- [STABLE] Optimize mix precision and shared parameter scenarios.
- [STABLE] Optimize distributed prediction scenarios.

#### Executor

- [STABLE] Support unified runtime in GPU and CPU backend.
- [STABLE] MindSpore GPU support CUDA11 with cuDNN8.
- [STABLE] MindSpore GPU inference performance optimization by integrating TensorRT.
- [STABLE] MindSpore built on one Linux distribution can now be used on multiple Linux distributions with the same CPU architecture (e.g. EulerOS, Ubuntu, CentOS).
- [STABLE] MindSpore Ascend support group convolution.

#### DataSet

- [STABLE] Support caching over MindRecord dataset.
- [STABLE] Support new shuffle mode for MindRecord dataset.
- [STABLE] Support a cropper tool for MindSpore Lite to allow the user to customize MindData binary file according to their script.
- [STABLE] Support share memory mechanism to optimize the multi-processing efficiency of GeneratorDataset/Map/Batch.
- [STABLE] Add features for the GNN dataset to support molecular dynamics simulation scenarios.

#### FederatedLearning

- [STABLE] Support Cross-device federated learning framework.
- [STABLE] Support FL-Server distributed networking including TCP and HTTP communication.
- [STABLE] Support FL-Server distributed federated aggregation，support autoscaling and fault tolerance.
- [STABLE] Develop FL-Client framework.
- [STABLE] Supports local differential privacy algorithms.
- [STABLE] MPC-based security aggregation algorithm.
- [STABLE] MindSpore Lite Device-side Inference & Training Interconnection with FL-Client.

#### Running Data Recorder

- [STABLE] Provide records of multi-stage computational graphs, memory allocation information and graph execution order when a "Launch kernel failed" occurs. (CPU)

#### GraphKernel Fusion

- [STABLE] Add options to control the optimization level.
- [STABLE] Enhance the generalization ability on GPU. GraphKernel is enabled by default in 40+ networks which cover the field of NLP, CV, Recommender, NAS and Audio. The result shows their throughput is significantly improved, and you are Recommended enabling GraphKernel in your network.

#### Debug

- [STABLE] Unified dump function.

### API Change

#### Backwards Incompatible Change

##### Python API

###### `mindspore.dataset.Dataset.device_que` interface removes unused parameter `prefetch_size`([!18973](https://gitee.com/mindspore/mindspore/pulls/18973))

Previously, we have a parameter `prefetch_size` in `device_que` to define the prefetch number of records ahead of the user's request. But indeed this parameter is never used which means it is an ineffective parameter. Therefore, we remove this parameter in 1.3.0 and users can set this configuration by [mindspore.dataset.config.set_prefetch_size](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size).

<table>
<tr>
<td style="text-align:center"> 1.2.1 </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
device_que(prefetch_size=None, send_epoch_end=True, create_data_info_queue=False)
```

</td>
<td>

```python
device_que(send_epoch_end=True, create_data_info_queue=False)
```

</td>
</tr>
</table>

###### `mindspore.nn.optim.thor` interface changes to lowercase `thor` and adds two parameters `enable_clip_grad` and `frequency`([!17212](https://gitee.com/mindspore/mindspore/pulls/17212))

The parameter `enable_clip_grad` is used for gradient clipping and another parameter `frequency` is used to control the update interval of second order information matrix.

<table>
<tr>
<td style="text-align:center"> 1.2.1 </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
THOR(net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32,
     use_nesterov=False, decay_filter=lambda x: x.name not in [], split_indices=None)
```

</td>
<td>

```python
thor(net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32,
     use_nesterov=False, decay_filter=lambda x: x.name not in [], split_indices=None, enable_clip_grad=False,
     frequency=100)
```

</td>
</tr>
</table>

##### Dump Config

Previously, we could only dump tensor data for one or all steps. To make the dump feature easier to use, we changed the dump configuration format and dump structure. View the [New Dump Tutorial](https://www.mindspore.cn/docs/programming_guide/en/r1.3/dump_in_graph_mode.html).

| 1.2.1                                                  | 1.3.0                                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `iteration` is an int.                               | `iteration` is a string.                                                                  |
| `op_debug_mode` is in `async_dump_settings` field. | `op_debug_mode` is in `common_dump_settings` field. `async_dump_settings` is removed. |

### Bug fixes

#### FrontEnd

- Fix exception when use import module in while body such as 'F.xxx'.([!17635](https://e.gitee.com/mind_spore/repos/mindspore/mindspore/pulls/17635))
- Fix the exception of 'exceeding limit call depth' in compile graph process when using while expression with grad operation. ([!18662](https://e.gitee.com/mind_spore/repos/mindspore/mindspore/pulls/18662))

#### Executor

- Fix reallocate memory bug for communication op.([!14492](https://gitee.com/mindspore/mindspore/pulls/14492))
- Replace memcpy_async op with tensor_move op.([!15204](https://gitee.com/mindspore/mindspore/pulls/15204))
- Fix the build error when multiple python versions are installed in the environment. ([!19165](https://gitee.com/mindspore/mindspore/pulls/19165))
- The warning when the te/topi/hccl version does not match is optimized, and fix the repeated warning. ([!18704](https://gitee.com/mindspore/mindspore/pulls/18704))
- Fix the error in a cluster with more than 8 pcs in pynative mode. ([!16376](https://gitee.com/mindspore/mindspore/pulls/16376))
- Fix graph ring problem in UB fusion.([!16109](https://gitee.com/mindspore/mindspore/pulls/16109))
- Fix AllGather op select problem when the shape is not divisible by 16. ([!18878](https://gitee.com/mindspore/mindspore/pulls/18878))

#### Dataset

- Fix an out-of-memory error when ImagefolderDataset gets an illegal directory. ([!16196](https://gitee.com/mindspore/mindspore/pulls/16196))
- Fix bugs of vision transformations in lite mode. ([!14722](https://gitee.com/mindspore/mindspore/pulls/14722),[!14774](https://gitee.com/mindspore/mindspore/pulls/14774),[!15050](https://gitee.com/mindspore/mindspore/pulls/15050))
- Fix default numbers of parallel workers of MindData for those CPUs with fewer cores. ([!15921](https://gitee.com/mindspore/mindspore/pulls/15921))
- Fix MindRecord writing failed probabilistically in multiprocessing. ([!15242](https://gitee.com/mindspore/mindspore/pulls/15242))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Support Caffe model running on Hi3516D.
2. Support delegate mechanism to run your models(part or whole) on user specified executor.
3. Support control flow models.
4. Support cross-compiling for iOS, so that we can inference models on iOS devices.

#### x86 backend optimization

1. Optimize kernels for x86 using Advanced Vector Extensions(AVX).

#### ARM backend optimization

1. Optimize fp16 kernels.
2. Support arm32 fp16 instruction acceleration on ARMv8.2.

#### Cuda backend optimization

1. Support NV GPU backend base on delegate mechanism(use TensorRT as delegate).

#### OpenCL backend

1. Optimize the strategy of workgroup and blocksize to improve performance.
2. Support OpenCL dynamic infershape.
3. Support INT32 type ops.

#### Post quantization

1. Support fp32 training model converts to quantization training model.

#### Training on Device

1. Support fp32 training model export to quantization model after training process end.
2. Unify APIs and output package name of training and inference.
3. Simplify implementation of Train Session.
4. Optimize train and infer compile, reduce libmindspore-lite-train.so memory.
5. Training memory optimization:  memory reduce 10-50% compare with  r1.2.
6. Training performance optimization:  for 1*1 special input shape Cov2DGradInput and SparseSoftmaxCrossEntropyWithLogits operator optimization, improved 10%-20%.
7. Support more networks(transformer, albert).

#### Codegen

1. Support deployment on HarmonyOS for device.

### API Change

#### API Incompatible Change

##### C++ API

###### Unify LiteSession and TrainSession, Merge LiteSession And TrainSession.([!17356](https://gitee.com/mindspore/mindspore/pulls/17356))

Previously, Training on Device use TrainSession while Inference on Device use LiteSession. To simplify implementation, we move TrainSession functions to LiteSession as virtual function. and move APIs previous defined in train_session.h to lite_session.h.

```cpp
class MS_API LiteSession {
...
static LiteSession *CreateTrainSession(const std::string &filename, const lite::Context *context,
                                         bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
 static LiteSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head,
                                            const lite::Context *context, bool train_mode = false,
                                            const lite::TrainCfg *cfg = nullptr);
virtual int Train() { return mindspore::lite::RET_ERROR; }
virtual int Eval() { return mindspore::lite::RET_OK; }
virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) {
    return mindspore::lite::RET_ERROR;
  }
virtual std::vector<tensor::MSTensor *> GetPredictions() const {
    std::vector<tensor::MSTensor *> outputs;
    return outputs;
 }
...
```

###### Add Export API for Training on device, obsolete SaveToFile API.([!17356](https://gitee.com/mindspore/mindspore/pulls/17356))

Previously, Training on Device uses SaveToFile API to save the training model to file. Export API was added in this release to support more format, more model type(train or interface part of the model), and save weight quant model of train.

```cpp
virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFERS) {
    return mindspore::lite::RET_ERROR;
 }
```

###### Add GetFeatureMaps and UpdateFeatureMaps interface for Training on device.([!18344](https://gitee.com/mindspore/mindspore/pulls/18344))

When Training on the device, we may need to update the model featuremap and get model featuremap.particularly in MindSpore Federated Scenario.

```cpp
virtual std::vector<tensor::MSTensor *> GetFeatureMaps() const {
    std::vector<tensor::MSTensor *> features;
    return features;
  }
  virtual int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features) { return mindspore::lite::RET_ERROR; }
```

#### New features

##### Java API

###### new static method for creating LiteSession by MSConifg in LiteSession.class

Previously, if we want to create a LiteSession object, we need to call two APIs:

```js
MSConfig config;
// config options ...
LiteSession liteSession = new LiteSession();
boolean ret = liteSession.init(config);
if (!ret) {
  // handle init LiteSession failed ...
}
```

now we can create a LiteSession object with new API just like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = createSession(config);
if (liteSession == null) {
  // handle create LiteSession failed ...
}
```

###### new static method for creating LiteSession byModelBuffer and MSConfig in LiteSession.class

Previously, if we want to inference a model, we need to call APIs like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = new LiteSession();
boolean initSessionRet = liteSession.init(config);
if (!initSessionRet) {
  // handle init LiteSession failed and return ...
}
Model model = new Model();
boolean loadModelRet = model.loadModel(modelMappedByteBuffer);
if (!loadModelRet) {
  // handle load model failed and return ...
}
boolean compileModelRet = liteSession.compileGraph(model);
if (!loadModelRet) {
  // handle compile model failed and return ...
}
model.free();
// liteSession is ready to inference model, call runGraph in LiteSession.class ...
```

now we can use new API just like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = createSession(modelMappedByteBuffer, config);
if (liteSession == null) {
  // handle init LiteSession failed and return ...
}
// liteSession is ready to inference model, call runGraph in LiteSession.class ...
```

New createSession method is an API that integrates four old APIs: LiteSession.init, Model.loadModel, LiteSession.compileGraph and model.free. It is simple and efficient as it reduces one modelBuffer copy operation.

###### new methods getFeaturesMap and updateFeatures for in LiteSession.class

Recently, we add a new C++ api in LiteSession class, Correspondingly we add a new java API in LiteSession.java.

```java
public List<MSTensor> getFeaturesMap() {
         List<Long> ret = this.getFeaturesMap(this.sessionPtr);
                ArrayList<MSTensor> tensors = new ArrayList<MSTensor>();
                for (Long msTensorAddr : ret) {
                    MSTensor msTensor = new MSTensor(msTensorAddr);
                    tensors.add(msTensor);
                }
                return tensors;
   }
   public boolean updateFeatures(List<MSTensor> features) {
            long[] inputsArray = new long[features.size()];
            for (int i = 0; i < features.size(); i++) {
                inputsArray[i] = features.get(i).getMSTensorPtr();
            }
             return this.updateFeatures(this.sessionPtr, inputsArray);
   }
```

###### new methods export to replace saveToFile API in LiteSession.class

Recently, we add a new C++ api in LiteSession class, Correspondingly we add a new java API in LiteSession.java.

```java
public boolean export(String modelFileName, int modelType, int quantizationType) {
        return this.export(this.sessionPtr, modelFileName, modelType, quantizationType);
    }
```

###### new train related  API moved to LiteSession.class from TrainSession.class

Align with update of C++ api in LiteSession class, add new java API to LiteSession.java Correspondingly.

```java
public class LiteSession {
...
public static LiteSession createTrainSession(String modelName, final MSConfig config, boolean trainMode){...}
public boolean train() {...}
public boolean eval() {...}
...
```

### Bug fixes

1. Fix the bug that the train session does not release memory cause of refcount bug.

#### Deprecations

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.2.1

## MindSpore 1.2.1 Release Notes

### Major Features and Improvements

#### FrontEnd

- [STABLE] Add MaskedSelect aicpu operation.(Ascend)

#### Auto Parallel

- [STABLE] Support distributed checkpoint loading.(Ascend/GPU)

# MindSpore 1.2.0

## MindSpore 1.2.0 Release Notes

### Major Features and Improvements

#### NewModels

- [STABLE] Add CV models on Ascend: 3D Unet, Unet++, SSD-Resnet50-fpn, SSD-VGG16, crnn_seq2seq_ocr for BSI, CTPN, resnet18, DPN
- [STABLE] Add CV models on GPU: Faster-RCNN
- [STABLE] Add NLP models on Ascend: NAML, Fasttext, GRU, LSTM
- [BETA] Add TPRR: Thinking Path Re-Ranker, an original ranked-base framework for Multi-Hop Question Answering which has won the first place in HotpotQA leaderboard.(Ascend)

#### FrontEnd

- [STABLE] Support side effects expression to ensure that the perform order of user's semantics is correct.(Ascend/GPU/CPU)
- [STABLE] Support calculating the gradient for network that contain non-Tensor input parameters（int, float, bool, mstype,int, mstype.float, mstype.uint, mstype.bool_, tuple, list, dict）.(Ascend/GPU/CPU)
- [STABLE] Support the inverse of a bool Tensor.(Ascend/GPU/CPU)
- [STABLE] Uniform the interface `isinstance`.(Ascend/GPU/CPU)
- [STABLE] Support negative indexes.(Ascend/GPU/CPU)
- [STABLE] Support 110+ Numpy-like interfaces in mindspore.numpy.(Ascend/GPU/CPU)
- [STABLE] Support export/load mindir model with a size greater than 2 GB.
- [STABLE] The optimizer supports gradient centralization.(Ascend)
- [STABLE] Support support auc metric, rou metric, bleu score metric, confusion matrix metric, cosine similarity metric, dice metric, hausdorff distance metric, occlusion sensitivity metric, perplexity metric, mean surface distance metric, root mean surface distance metric.
- [STABLE] Support use EmbeddingLookup with cache.(Ascend)
- [STABLE] Add MaskedSelect aicpu operation.(Ascend)

#### Auto Parallel

- [STABLE] Support AllGather and ReduceScatter fusion.(Ascend)
- [STABLE] Support gradient accumulation feature in auto parallel mode.(Ascend/GPU)
- [STABLE] Support running parallel optimizer with gradient accumulation.(Ascend)
- [STABLE] Add the configuration of communication operators' fusion.(Ascend)
- [STABLE] Support distributed checkpoint loading.(Ascend/GPU)

#### Executor

- [STABLE] Support inference with Nvidia GPU.
- [STABLE] Support data parallelism in PyNative mode.(Ascend/GPU)
- [STABLE] Optimize LSTM inference memory consumption in Graph mode with CPU.

#### Sponge

- [STABLE] Add SPONGE modules for molecular dynamics simulation, including Bond, Angle, Dihedral, Non Bond 14, NeighborList, Particle Mesh Ewald, Langevin MD and LIUJIAN MD.(GPU)

#### DataSet

- [STABLE] If the libnuma library is installed in the environment, you can run `export DATASET_ENABLE_NUMA=True` or `export MS_ENABLE_NUMA=True` to configure NUMA binding. In multi-card training scenarios, the training data processing speed can be improved, thereby improving the network training efficiency.
- [STABLE] Unify API Tensor structure of Training/Inference interfaces in C++ SDK.
- [STABLE] Optimize duplicated Decode in data preprocess using cache, improve preprocess efficiency.
- [STABLE] Support eager mode to run data augmentation in Python & C++.
- [STABLE] Support more data augmentation operators(e.g. Affine, Perspective) in MindSpore-Lite.
- [STABLE] Support light pipeline to process MindData in MindSpore-Lite training.
- [STABLE] Support more data preprossing operators based on DVPP hardware module and can be used on on Ascend310 platform.
- [STABLE] Support copy-free property for data in Ascend310 inference process scenarios.

#### Running Data Recorder

- [STABLE] Support running data recorder (RDR)  for exception demarcation.
- [STABLE] Provide records of multi-stage computational graphs, memory allocation information, graph execution order, stream execution order and task debug information when a "run task error" or "distribute task failed" occurs. (Ascend)
- [STABLE] Provide records of multi-stage computational graphs, memory allocation information and graph execution order when a "SyncStream error" occurs. (GPU)

#### 3D Feature

- [STABLE] Support 3D ops: Conv3D, Conv3DBackpropInput, Conv3DBackpropFilter, Conv3DTranspose, BiasAdd, BiasAddGrad, PReLU, Transpose, Reshape, transdata, StrideSlice, MaxPool3D, MaxPool3DGrad, BinaryCrossEntropy, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsGrad, SoftmaxCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsGrad, BatchNorm3d, BatchNorm3dGrad, Dropout3d.
- [STABLE] Support RMSELoss loss function, MAELoss loss function, FocalLoss loss function, DiceLoss binary loss function, and MultiClassDiceLoss multi-type loss function for 2D/3D network.
- [STABLE] Add optimizer: AdamApplyOne(3D), ApplyMomentum(3D), SGD(3D).

### API Change

#### Backwards Incompatible Change

##### Python API

###### `mindspore.numpy.array()`, `mindspore.numpy.asarray()`, `mindspore.numpy.asfarray()`, `mindspore.numpy.copy()` now support GRAPH mode, but cannot accept `numpy.ndarray` as input arguments anymore([!12726](https://gitee.com/mindspore/mindspore/pulls/12726))

Previously, these interfaces can accept numpy.ndarray as arguments and convert numpy.ndarray to Tensor, but cannot be used in GRAPH mode.
However, currently MindSpore Parser cannot parse numpy.ndarray in JIT-graph. To support these interfaces in graph mode, we have to remove `numpy.ndarray` support. With that being said, users can still use `Tensor` to convert `numpy.ndarray` to tensors.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.numpy as mnp
>>> import numpy
>>>
>>> nd_array = numpy.array([1,2,3])
>>> tensor = mnp.asarray(nd_array) # this line cannot be parsed in GRAPH mode
```

</td>
<td>

```python
>>> import mindspore.numpy as mnp
>>> import numpy
>>>
>>> tensor = mnp.asarray([1,2,3]) # this line can be parsed in GRAPH mode
```

</td>
</tr>
</table>

###### mindspore.numpy interfaces remove support for keyword arguments `out` and `where`([!12726](https://gitee.com/mindspore/mindspore/pulls/12726))

Previously, we have incomplete support for keyword arguments `out` and `where` in mindspore.numpy interfaces, however, the `out` argument is only functional when `where` argument is also provided, and `out` cannot be used to pass reference to numpy functions. Therefore, we have removed these two arguments to avoid any confusion users may have. Their original functionality can be found in [np.where](https://www.mindspore.cn/docs/en/master/api_python/numpy/mindspore.numpy.where.html#mindspore.numpy.where)

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.numpy as np
>>>
>>> a = np.ones((3,3))
>>> b = np.ones((3,3))
>>> out = np.zeros((3,3))
>>> where = np.asarray([[True, False, True],[False, False, True],[True, True, True]])
>>> res = np.add(a, b, out=out, where=where) # `out` cannot be used as a reference, therefore it is misleading
```

</td>
<td>

```python
>>> import mindspore.numpy as np
>>>
>>> a = np.ones((3,3))
>>> b = np.ones((3,3))
>>> out = np.zeros((3,3))
>>> where = np.asarray([[True, False, True],[False, False, True],[True, True, True]])
>>> res = np.add(a, b)
>>> out = np.where(where, x=res, y=out) # instead of np.add(a, b, out=out, where=where)
```

</td>
</tr>
</table>

###### Turn `ops.MakeRefKey` into an internal interface ([!12010](https://gitee.com/mindspore/mindspore/pulls/12010))

Previously MakeRefKey is an external interface that is not used, now make it an internal interface with the same usage. We do not recommend users to use this interface, and we will remove the relevant introduction of this interface from the official website.

###### `ops.ApplyFtrl`, `ops.ApplyMomentum`, `ops.ApplyRMSProp`, `ops.ApplyCenteredRMSProp` change the output on Ascend backend from multiple to a single. ([!11895](https://gitee.com/mindspore/mindspore/pulls/11895))

Previously the number of outputs of these operator is different on different backends. To unify their definition we change their output on Ascend backend from multiple to a single.

##### `P.FusedBatchNorm`, `P.FusedBatchNormEx` deleted ([!12115](https://gitee.com/mindspore/mindspore/pulls/12115))

The FusedBatchNorm and FusedBatchNormEx interface has been deleted. Please use the batchnorm operator to replace it.

##### `MetaTensor` deleted ([!10325](https://gitee.com/mindspore/mindspore/pulls/10325))

The MetaTensor interface has been deleted. The function of MetaTensor has been integrated into tensor.

###### `ControlDepend` is deleted, use `Depend` instead. The decorator `@C.add_flags(has_effect=True)` does not work. ([!13793](https://gitee.com/mindspore/mindspore/pulls/13793))

Previously, we used ControlDepend to control the execution order of multiple operators. In version 1.2.0, mindspore introduces the auto-monad side effects expression to ensure that the perform order of user's semantics is correct. Therefore, ControlDepend is deleted and Depend is recommended.

In most scenarios, if operators have IO side effects (such as print) or memory side effects (such as assign), they will be executed according to the user's semantics. In some scenarios, if the two operators A and B have no order dependency, and A must be executed before B, we recommend using Depend to specify their execution order. See the API documentation of the Depend operator for specific usage.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
    In some side-effect scenarios, we need to ensure the execution order of operators.
    In order to ensure that operator A is executed before operator B, it is recommended
    to insert the Depend operator between operators A and B.

    Previously, the ControlDepend operator was used to control the execution order.
    Since the ControlDepend operator is deprecated from version 1.1, it is recommended
    to use the Depend operator instead. The replacement method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
        ControlDepend(a, b)     --->        b = B(y)
```

</td>
<td>

```python
    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
                                --->        b = B(y)
```

</td>
</tr>
</table>

After the introduction of the auto-monad side effect expression feature, the decorator `@C.add_flags(has_effect=True)` does not work. If the decorator is used in the script, please modify. Take the overflow identification operator (without side effects) as an example, the modification method is as follows:

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
@C.add_flags(has_effect=True)
def construct(self, *inputs):
    ...
    loss = self.network(*inputs)
    init = self.allo_status()
    self.clear_status(init)
    ...
```

</td>
<td>

```python
def construct(self, *inputs):
    ...
    loss = self.network(*inputs)
    init = self.allo_status()
    init = F.depend(init, loss)
    clear_status = self.clear_status(init)
    ...
```

</td>
</tr>
</table>

##### C++ API

###### C++ API support dual ABI now.([!12432](https://gitee.com/mindspore/mindspore/pulls/12432))

1.1.1 supports only the old ABI. Currently, both the new and the old are supported.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
```

</td>
<td>

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)  # old ABI are supported
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)  # new ABI are supprrted, too
                                                   # write nothing, use new ABI as default
```

</td>
</tr>
</table>

###### Context refactor.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

The `Context` class is refactored. For details, see the API docs.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
GlobalContext::SetGlobalDeviceTarget(kDeviceTypeAscend310);       // set device target is ascend310
GlobalContext::SetGlobalDeviceID(0);                              // set device id is 0
auto model_context = std::make_shared<ModelContext>();            // create a model context
ModelContext::SetInsertOpConfigPath(model_context, "./aipp.cfg")  // set aipp config file is ./aipp.cfg
```

</td>
<td>

```cpp
auto model_context = std::make_shared<Context>();                 // create a model context
auto ascend310_info = std::make_shared<Ascend310DeviceInfo>();
model_context.MutableDeviceInfo().push_back(ascend310_info );     // set device target is ascend310
ascend310_info->SetDeviceID(0);                                   // set device id is 0
ascend310_info->SetInsertOpConfigPath("./aipp.cfg");              // set aipp config file is ./aipp.cfg
```

</td>
</tr>
</table>

###### LoadModel interface changes.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`LoadModel` is renamed `Load`. No exception is thrown new but the return status should be checked.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
try {
  auto graph = Serialization::LoadModel(model_file_path, kMindIR);
} catch (...) { ... }
```

</td>
<td>

```cpp
Graph graph;
auto ret = Serialization::Load(model_file_path, kMindIR, &graph);
if (ret != kSuccess) { ... }
```

</td>
</tr>
</table>

###### Model ctor changes.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`Model` uses a non-parameter ctor now, and arguments are passed in through `Build`.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
Model net(net_cell, model_context);
auto ret = net.Build();
if (ret != kSuccess) { ... }
```

</td>
<td>

```cpp
Model net;
auto ret = net.Build(net_cell, model_context);
if (ret != kSuccess) { ... }
```

</td>
</tr>
</table>

###### MSTensor::CreateTensor returns a native pointer now.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`MSTensor::CreateTensor` and `MSTensor::CreateRefTensor` returns a native pointer now, need to be destroy by `DestroyTensorPtr`.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
auto tensor = MSTensor::CreateTensor(xxx, xxx, ...);
auto name = tensor.Name();
```

</td>
<td>

```cpp
auto tensor = MSTensor::CreateTensor(xxx, xxx, ...);
auto name = tensor->Name();
MSTensor::DestroyTensorPtr(tensor);
```

</td>
</tr>
</table>

#### New features

##### Python API

- Add SPONGE functions: `mindspore.ops.operations.BondForceWithAtomEnergy`, `mindspore.ops.operations.AngleForceWithAtomEnergy`, `mindspore.ops.operations.DihedralForceWithAtomEnergy`, `mindspore.ops.operations.Dihedral14LJCFForceWithAtomEnergy`, `mindspore.ops.operations.LJForceWithPMEDirectForce`, `mindspore.ops.operations.PMEExcludedForce`, `mindspore.ops.operations.PMEReciprocalForce`,`mindspore.ops.operations.BondEnergy`, `mindspore.ops.operations.AngleEnergy`,`mindspore.ops.operations.DihedralEnergy`, `mindspore.ops.operations.Dihedral14LJEnergy`, `mindspore.ops.operations.Dihedral14CFEnergy`,`mindspore.ops.operations.LJEnergy`, `mindspore.ops.operations.PMEEnergy`. All operators are supported in `GPU`.

#### Deprecations

##### Python API

###### `nn.MatMul` is now deprecated in favor of `ops.matmul` ([!12817](https://gitee.com/mindspore/mindspore/pulls/12817))

[ops.matmul](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.matmul.html#mindspore.ops.matmul) follows the API of [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) as closely as possible. As a function interface, [ops.matmul](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.matmul.html#mindspore.ops.matmul) is applied without instantiation, as opposed to `nn.MatMul`, which should only be used as a class instance.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import numpy as np
>>> from mindspore import Tensor, nn
>>>
>>> x = Tensor(np.ones((2, 3)).astype(onp.float32)
>>> y = Tensor(np.ones((3, 4)).astype(onp.float32)
>>> nn.MatMul()(x, y)
```

</td>
<td>

```python
>>> import numpy as np
>>> from mindspore import Tensor, ops
>>>
>>> x = Tensor(np.ones((2, 3)).astype(onp.float32)
>>> y = Tensor(np.ones((3, 4)).astype(onp.float32)
>>> ops.matmul(x, y)
```

</td>
</tr>
</table>

### Bug fixes

#### FrontEnd

- fix the null pointer problem of evaluator in control flow.([!13312](https://gitee.com/mindspore/mindspore/pulls/13312))
- fix parameter naming conflict bug for CellList and SequentialCell. ([!13260](https://gitee.com/mindspore/mindspore/pulls/13260))

#### Executor

- fix executor pending task not execute in some heterogeneous cases.([!13465](https://gitee.com/mindspore/mindspore/pulls/13465))
- add passes to support frontend IR unification, including following operations: SliceGrad([!11783](https://gitee.com/mindspore/mindspore/pulls/11783)), ApplyFtrl, ApplyMomentum, ApplyRMSProp, CenteredRMSProp([!11895](https://gitee.com/mindspore/mindspore/pulls/11895)), AvgPoolGrad([!12813](https://gitee.com/mindspore/mindspore/pulls/12813)), BatchNorm([!12115](https://gitee.com/mindspore/mindspore/pulls/12115))

#### Dataset

- Fix getter functions(e.g. GetDatasetSize) terminated abnormally when use python multi-processing. ([!13571](https://gitee.com/mindspore/mindspore/pulls/13571), [!13823](https://gitee.com/mindspore/mindspore/pulls/13823))
- Fix unclear error log of data augmentation operators. ([!12398](https://gitee.com/mindspore/mindspore/pulls/12398), [!12883](https://gitee.com/mindspore/mindspore/pulls/12883), [!13176](https://gitee.com/mindspore/mindspore/pulls/13176))
- Fix profiling performs abnormally when sink_size = False, as saving data is later than profiling analysis. ([!13944](https://gitee.com/mindspore/mindspore/pulls/13944))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Support TensorFlow model in Converter except aware-training model.
2. Add fusion pattern for same horizontal operators in Converter.
3. Support Jar in x86_64 system for integrating into server with Java backend conveniently.
4. Provide unified runtime API for developer reusing their code between cloud side and end side.[BETA]
5. Improve control-flow capabilities continually: Support GRU fusion in Converter; Support weight-quant for control-flow model; Support control-flow model inference with half precision; Support nested control-flow model.[BETA]

#### ARM backend optimization

1. Add NLP dependent float16 operators(like lstm) to enhance inference performance.
2. Optimize operators: lstm, gru, depthwise.
3. Add 6 NPU operators(like FullConnection), and fix some bugs about buildIR failed.

#### OpenCL backend

1. Add new ops: add 10+ ops，total 72 ops；
2. Performance optimization: by memory layout optimize，block tiling，Performance improved by 30% compared to version 1.1 at Adreno GPU.
3. Initialization time optimization: initialization time improve 100% vs MSLITE Version1.1 by store kernel cache as binary.
4. Support Java call on Mali or Adreno GPU.

#### Post quantization

1. Support quantization of gather and lstm ops.
2. Support quantizatizing TF Lite models with sub-graph node.
3. Add quantiztion strategy to decide quantize ops or not，less accuracy loss and higher compression rate.

#### Training on Device

1. Virtual batching, use mini-batch to minic large batch in theorical with few RAM consumption.
2. Converter unify, do not compile tod and iod converter separately.
3. Performance optimization to BWD ops.
4. TrainLoop with Off-The-Shelf Functionality blocks, like LR scheduler, Loss Monitor, Ckpt Saver, Accuracy Monitor.
5. Integration of code with Minddata lite.
6. Support more networks (googlenet, densenet, shufflenetv2, nin, vgg) and operators.

#### Codegen

1. Support 79 ops for the ARM platform and all CMSIS ops for Arm Cortex-M Series.
2. Multiplatform support, including Android, IoT Devices.
3. Support offline model weight preprocessing while compiling.
4. Support offline memory reuse computing for minimum runtime buffer size.
5. Support kernel register for custom op. Third-party hardware like NNIE can be accessed through it.

### API Change

#### API Incompatible Change

##### C++ API

###### Add header file named lite_types.h for some common data structs. ([!12262](https://gitee.com/mindspore/mindspore/pulls/12262))

Previously, some common data structs such as `CpuBindMode` and `DeviceType` are in context.h, this may cause cross-dependency between headers. So we create a new header named lite_types.h for some common data structs and move `CpuBindMode` and `DeviceType` from context.h into lite_types.h.

<table>
<tr>
<td style="text-align:center"> lite_types.h </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND,    /**< no bind */
  HIGHER_CPU, /**< bind higher cpu first */
  MID_CPU     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type */
} DeviceType;
}  // namespace mindspore::lite
```

</td>
</tr>
</table>

###### Add some new interfaces in ms_tensor.h for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, users could not create `MSTensor` or modify ``MSTensor, all `MSTensor` are created and managed by framework. However users need to create or modify MSTensor sometimes such as pre-processing input data. So we provide two new interfaces in ms_tensor.h: `CreateTensor` interface for creating `MSTensor` by user and `set_shape` interface for modifying the shape of `MSTensor`.

<table>
<tr>
<td style="text-align:center"> CreateTensor </td>
</tr>
<tr>
<td>

```cpp
/// \brief Create a MSTensor.
///
/// \return Pointer to an instance of MindSpore Lite MSTensor.
static MSTensor *CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape, const void *data,
                                size_t data_len);
```

</td>
</tr>
</table>

<table>
<tr>
<td style="text-align:center"> set_shape </td>
</tr>
<tr>
<td>

```cpp
/// \brief Set the shape of MSTensor.
virtual void set_shape(const std::vector<int> &shape) = 0;
```

</td>
</tr>
</table>

Previously, users could access to data of `MSTensor` by interface named `MutableData`. However `MutableData` is not only returning data of tensor but also allocating data for tensor if its data is nullptr. So we provide a new interfaces in ms_tensor.h named `data` for returning data of tensor without allocating automatically.

<table>
<tr>
<td style="text-align:center"> data </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get the pointer of data in MSTensor.
///
/// \note The data pointer can be used to both write and read data in MSTensor. No memory buffer will be
/// allocated.
///
/// \return the pointer points to data in MSTensor.
virtual void *data() = 0;
```

</td>
</tr>
</table>

###### Delete `DimensionSize()` in ms_tensor.h.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

The interface named `DimensionSize` is fuinctionally overlapped with the interface named `shape`. For the simplicity of the interface, we delete `DimensionSize` and recommend users to use the new interface named `shape` instead.

<table>
<tr>
<td style="text-align:center"> DimensionSize() </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
///
/// \param[in] index Define index of dimension returned.
///
/// \return Size of dimension of the MindSpore Lite MSTensor.
virtual int DimensionSize(size_t index) const = 0;
```

</td>
</tr>
</table>

###### Move allocator from namespace mindspore::lite to namespace lite for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, class `Allocator` is in namespace mindspore::lite. Considering unified allocator interface for unified runtime API, we move `Allocator` to namespace mindspore.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
<td>

```cpp
namespace mindspore {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
</tr>
</table>

### Bug fixes

1. Fix the bug that the array in kernel registrar is not initialized.
2. Fix segment fault caused by releasing of OpParameter in Crop kernel in mistake.
3. Fix the bug that the MINDIR aware-training model is finally interpreted as weight-quant model.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa.

Contributions of any kind are welcome!

# MindSpore 1.1.1 Release Notes

## MindSpore

### Major Features and Improvements

#### NewModels

- [STABLE] BGCF: a Bayesian Graph Collaborative Filtering(BGCF) framework used to model the uncertainty in the user-item interaction graph and thus recommend accurate and diverse items on Amazon recommendation dataset.(Ascend)
- [STABLE] GRU: a recurrent neural network architecture like the LSTM(Long-Short Term Memory) on Multi30K dataset.(Ascend)
- [STABLE] FastText: a simple and efficient text classification algorithm on AG's news topic classification dataset, DBPedia Ontology classification dataset and Yelp Review Polarity dataset.(Ascend)
- [STABLE] LSTM: a recurrent neural network architecture used to learn word vectors for sentiment analysis on aclImdb_v1 dataset.(Ascend)
- [STABLE] SimplePoseNet: a convolution-based neural network for the task of human pose estimation and tracking on COCO2017 dataset.(Ascend)

#### FrontEnd

- [BETA] Support Tensor Fancy Index Getitem with tuple and list. (Ascend/GPU/CPU)

### Backwards Incompatible Change

#### Python API

##### `ops.AvgPool`, `ops.MaxPool`, `ops.MaxPoolWithArgmax` change attr name from 'ksize', 'padding' to 'kernel_size', 'pad_mode' ([!11350](https://gitee.com/mindspore/mindspore/pulls/11350))

Previously the kernel size and pad mode attrs of pooling ops are named "ksize" and "padding", which is a little puzzling and inconsistent with convolution ops. So they are rename to "kernel_size" and "pad_mode".

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(ksize=2, padding='same')
>>> max_pool = ops.MaxPool(ksize=2, padding='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(ksize=2, padding='same')
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(kernel_size=2, pad_mode='same')
>>> max_pool = ops.MaxPool(kernel_size=2, pad_mode='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(kernel_size=2, pad_mode='same')
```

</td>
</tr>
</table>

##### `ops.TensorAdd`, change API name to `ops.Add` ([!11568](https://gitee.com/mindspore/mindspore/pulls/11568))

The operator name TensorAdd is not standardized, it is changed to Add. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.TensorAdd()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.Add()
```

</td>
</tr>
</table>

##### `ops.Gelu`, `ops.GeluGrad`, `ops.FastGelu`, `ops.FastGeluGrad`, change API name to `ops.GeLU`, `ops.GeLUGrad`, `ops.FastGeLU`, `ops.FastGeLUGrad` ([!11603](https://gitee.com/mindspore/mindspore/pulls/11603))

Gelu, GeluGrad, FastGelu, and FastGeluGrad names are unified into ReLU naming rules, "lu" is changed to the uppercase "LU". The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.Gelu()
>>> gelu_grad = ops.GeluGrad()
>>> fast_gelu = ops.FastGelu()
>>> fast_gelu_grad = ops.FastGeluGrad()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.GeLU()
>>> gelu_grad = ops.GeLUGrad()
>>> fast_gelu = ops.FastGeLU()
>>> fast_gelu_grad = ops.FastGeLUGrad()
```

</td>
</tr>
</table>

##### `ops.GatherV2`, change API name to `ops.Gather` ([!11713](https://gitee.com/mindspore/mindspore/pulls/11713))

GatherV2 is changed to Gather. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.GatherV2()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.Gather()
```

</td>
</tr>
</table>

##### `ops.Pack`、`ops.Unpack`, change API name to `ops.Stack`、`ops.Unstack` ([!11828](https://gitee.com/mindspore/mindspore/pulls/11828))

Pack is changed to Stack, and Unpack is changed to Unstack. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> pack= ops.Pack()
>>> unpack= ops.Unpack()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> stack= ops.Stack()
>>> unstack= ops.Unstack()
```

</td>
</tr>
</table>

##### `ops.ControlDepend`, add deprecated to ControlDepend ([!11844](https://gitee.com/mindspore/mindspore/pulls/11844))

ControlDepend is deprecated and will be removed in a future version, use Depend instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```pythonNote:
Note:
    This operation does not work in `PYNATIVE_MODE`.
```

</td>
<td>

```python
Note:
        This operation does not work in `PYNATIVE_MODE`.
        `ControlDepend` is deprecated from version 1.1 and will be removed in a future version, use `Depend` instead.
```

</td>
</tr>
</table>

##### `ops.Depend`, add operator description and use case ([!11815](https://gitee.com/mindspore/mindspore/pulls/11815)), ([!11879](https://gitee.com/mindspore/mindspore/pulls/11879))

Since the ControlDepend operator will be deprecated from version 1.2, it is recommended to use the Depend operator instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
Depend is used for processing side-effect operations.

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``
```

</td>
<td>

```python
Depend is used for processing dependency operations.

In some side-effect scenarios, we need to ensure the execution order of operators.
In order to ensure that operator A is executed before operator B, it is recommended
to insert the Depend operator between operators A and B.

Previously, the ControlDepend operator was used to control the execution order.
Since the ControlDepend operator will be deprecated from version 1.2, it is
recommended to use the Depend operator instead. The replacement method is as follows::

    a = A(x)                --->        a = A(x)
    b = B(y)                --->        y = Depend(y, a)
    ControlDepend(a, b)     --->        b = B(y)

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

Examples:
    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.ops.operations as P
    >>> from mindspore import Tensor
    >>> class Net(nn.Cell):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.softmax = P.Softmax()
    ...         self.depend = P.Depend()
    ...
    ...     def construct(self, x, y):
    ...         mul = x - y
    ...         y = self.depend(y, mul)
    ...         ret = self.softmax(y)
    ...         return ret
    ...
    >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> net = Net()
    >>> output = net(x, y)
    >>> print(output)
    [[0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]]
```

</td>
</tr>
</table>

#### C++ API

##### change namespace from `mindspore::api` to `mindspore` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
namespace ms = mindspore::api;
```

</td>
<td>

```c++
namespace ms = mindspore;
```

</td>
</tr>
</table>

##### `Context` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Context::Instance().SetDeviceTarget(ms::kDeviceTypeAscend310).SetDeviceID(0);
```

</td>
<td>

```c++
ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend310);
ms::GlobalContext::SetGlobalDeviceID(0);
```

</td>
</tr>
</table>

##### rename `Tensor` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Tensor a;
```

</td>
<td>

```c++
ms::MSTensor a;
```

</td>
</tr>
</table>

##### `Model` move setting of model options from `Build` to ctor `Model` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Model model(graph_cell);
model.Build(model_options);
```

</td>
<td>

```c++
ms::Model model(graph_cell, model_context);
model.Build();
```

</td>
</tr>
</table>

##### `Model` modify `GetInputsInfo`, `GetOutputsInfo` to `GetInputs`, `GetOutputs` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<std::string> names;
std::vector<ms::DataType> types;
std::vector<std::vector<int64_t>> shapes;
std::vector<size_t> mem_sizes;
model.GetInputsInfo(&names, &types, &shapes, &mem_sizes);
std::cout << "Input 0 name: " << names[0] << std::endl;
```

</td>
<td>

```c++
auto inputs = model.GetInputs();
std::cout << "Input 0 name: " << inputs[0].Name() << std::endl;
```

</td>
</tr>
</table>

##### `Model` modify `Predict` parameters type from `Buffer` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<ms::Buffer> inputs;
std::vector<ms::Buffer> outputs;
model.Predict(inputs, &outputs);
```

</td>
<td>

```c++
std::vector<ms::MSTensor> inputs;
std::vector<ms::MSTensor> outputs;
model.Predict(inputs, &outputs);
```

</td>
</tr>
</table>

### Deprecations

#### Python API

##### `ops.SpaceToBatch`, `ops.BatchToSpace` are deprecated in favor of `ops.SpaceToBatchND`, `ops.BatchToSpaceND`([!11527](https://gitee.com/mindspore/mindspore/pulls/11527))

The `ops.SpaceToBatchND`, `ops.BatchToSpaceND` are more general and have same behavior as `ops.SpaceToBatch`, `ops.BatchToSpace` when `block_shape` is a int.

##### `ops.DepthwiseConv2dNative` is deprecated in favor of `nn.Conv2D`([!11702](https://gitee.com/mindspore/mindspore/pulls/11702))

The `ops.DepthwiseConv2dNative` is only supported by Ascend, it is recommended to directly use `nn.Conv2D`. If `group` is equal to `in_ channels` and `out_channels`, the 2D convolution layer is also a 2D depthwise convolution layer.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

# MindSpore 1.1.0 Release Notes

## MindSpore

### Major Features and Improvements

#### NewModels

- [STABLE] GNMT v2: similar to the model described in Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, which is mainly used for corpus translation, on WMT Englis-German dataset.(Ascend)
- [STABLE] MaskRCNN: a conceptually simple, flexible, and general framework for object instance segmentation on COCO2017 dataset.(Ascend)
- [STABLE] YOLOv4: a state-of-the-art detector which is faster and more accurate than all available alternative detectors on MS COCO dataset.(Ascend)
- [STABLE] Openpose: proposes a bottom-up human attitude estimation algorithm using Part Affinity Fields on COCO2017 dataset.(Ascend)
- [STABLE] CNN-CTC: proposes three major contributions to addresses scene text recognition (STR) on MJSynth and SynthText dataset.(Ascend)
- [STABLE] CenterFace: a practical anchor-free face detection and alignment method for edge devices on WiderFace dataset.(Ascend)
- [STABLE] ShuffleNetV2:  a much faster and more accurate network than the previous networks on ImageNet 2012 dataset.(GPU)
- [STABLE] EfficientNet-B0: a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient on ImageNet 2012 dataset.(GPU)
- [BETA] SSD-GhostNet: based on an Ghost module structure which generate more features from cheap operations on Oxford-IIIT Pet dataset.(Ascend)
- [BETA] DS-CNN:  Depthwise separable convolutional neural network on Speech commands dataset.(Ascend)
- [BETA] DeepPotentialH2O: A neural network model for molecular dynamics simulations. (Ascend)
- [BETA] GOMO: A classical numerical method called GOMO for ocean simulation. (GPU)

#### FrontEnd

- [STABLE] Refactor the MINDIR to support 310 inference(Ascend).
- [STABLE] The execution backend of sparse operations in optimizer can be set through 'target'. (Ascend/GPU/CPU)
- [STABLE] Support saving specified network to checkpoint and filtering parameters according to prefix when load checkpoint. (Ascend/GPU/CPU)
- [STABLE] Allow users choose whether to load parameter into network strictly.(Ascend/GPU/CPU)
- [STABLE] Before training, in graph mode, in order to have the same network initialization parameter values ​​for all devices, broadcast the parameters on device 0 to other devices. (Ascend/GPU)
- [STABLE] Support if by if of control flow subgraph. (Ascend/GPU)
- [STABLE] Support the judgment that whether a tensor is in a list. (Ascend/GPU/CPU)
- [STABLE] Support to get a value by using the corresponding key in a dictionary in the network; Support to get keys and values of a dictionary in the network. (Ascend/GPU/CPU)
- [STABLE] Support Tensor in enumerate. (Ascend/GPU/CPU)
- [STABLE] Support multilevel index assignment. (Ascend/GPU/CPU)
- [STABLE] Support the 'expand_as','view','abs','mean' method of Tensor. (Ascend/GPU/CPU)
- [STABLE] Support ResizeBilinear operation transfer ratio. (Ascend)
- [STABLE] nn.Matmul supports matrix-vector product and  batched matrix multiply. (Ascend/GPU)
- [STABLE] nn.Dense supports input tensor whose dimension can be greater than 2. (Ascend/GPU)
- [BETA] Support higher order differentiation for partial operators.(CPU/GPU/Ascend)
- [STABLE] Support Tensor Augassign.(Ascend/GPU)
- [BETA] Support 22 numpy native interfaces.

#### Auto Parallel

- [STABLE] Support parallel optimizer with weight shard. (Ascend/GPU)
- [STABLE] Support distributed operators: element-wise series, UnsortedSegmentSum, UnsortedSegmentMin, Split, BroadcastTo and Unique etc. (Ascend/GPU)
- [STABLE] Support distributed model prediction. (Ascend/GPU)
- [STABLE] Support auto mixed precision level "O2" in auto and semi auto parallel mode. (Ascend/GPU)
- [STABLE] Add MultiFieldEmbeddingLookup high-level interface. (Ascend/GPU)

#### Executor

- [STABLE] ResNet50 performance optimize. (GPU)
- [STABLE] Support modelzoo net in PyNative mode(Ascend 29, GPU 23, CPU 2).(Ascend/GPU/CPU)
- [STABLE] Support PyNative mode on CPU.(CPU)
- [STABLE] Optimize performance in PyNative mode.(Ascend/GPU/CPU)
- [STABLE] Support Safe Optimized Memory Allocation Solver (SOMAS) on Ascend to improve the memory-reuse, the batch size of Bert large model (128 sequence length) is increased from 160 to 208.(Ascend)
- [BETA] Support second order differentiation in PyNative mode.(Ascend/GPU)
- [DEMO] Add distributed trainning in PyNative mode.(Ascend/GPU)

#### MDP

- [STABLE]  Add new operators for Ascend and GPU: IGamma, LGamma, DiGamma;
- [STABLE]  Add new distributions for Ascend and GPU: LogNormal, and Logistic;
- [BETA]  Add new distributions for Ascend only: Gumbel, Cauchy, Gamma, Beta, and Poisson; Add Categorical distribution for GPU;
- [STABLE]  Add new bijectors for Ascend and GPU: GumbelCDF, Invert;
- [STABLE]  Add Bayesian layer realized by local reparameterization method for Ascend and GPU;
- [STABLE]  Add Anomaly Detection Toolbox based on VAE for Ascend and GPU.

#### DataSet

- [STABLE] Support single node multi-p distributed cache data sharing
- [STABLE] Support GPU profiling with data processing
- [STABLE] Support YOLOV3 dynamic shape in sink mode with dataset
- [STABLE] Support unique processing in the data processing pipeline
- [STABLE] Python layer parameter verification error information unified

### API Change

#### Backwards Incompatible Change

##### Python API

###### Delete shape and dtype of class Initializer ([!7373](https://gitee.com/mindspore/mindspore/pulls/7373/files))

Delete shape and dtype attributes of Initializer class.

###### Modify the return type of initializer ([!7373](https://gitee.com/mindspore/mindspore/pulls/7373/files))

Previously, the return type of initializer function may be string, number, instance of class Tensor or subclass of class Initializer.

After modification, initializer function will return instance of class MetaTensor, class Tensor or subclass of class Initializer.

Noted that the MetaTensor is forbidden to initialize parameters, so we recommend that use str, number or subclass of Initializer for parameters initialization rather than the initializer functions.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore import dtype as mstype
>>>
>>> def conv3x3(in_channels, out_channels)
>>>   weight = initializer('XavierUniform', shape=(3, 2, 32, 32), dtype=mstype.float32)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init=weight, has_bias=False, pad_mode="same")
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> #1) using string
>>> def conv3x3(in_channels, out_channels)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init='XavierUniform', has_bias=False, pad_mode="same")
>>>
>>> #2) using subclass of class Initializer
>>> def conv3x3(in_channels, out_channels)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init=XavierUniform(), has_bias=False, pad_mode="same")
```

</td>
</tr>
</table>

Advantages:
After modification, we can use the same instance of Initializer to initialize parameters of different shapes, which was not allowed before.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> weight_init_1 = XavierUniform(gain=1.1)
>>> conv1 = nn.Conv2d(3, 6, weight_init=weight_init_1)
>>> weight_init_2 = XavierUniform(gain=1.1)
>>> conv2 = nn.Conv2d(6, 10, weight_init=weight_init_2)
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> weight_init = XavierUniform(gain=1.1)
>>> conv1 = nn.Conv2d(3, 6, weight_init=weight_init)
>>> conv2 = nn.Conv2d(6, 10, weight_init=weight_init)
```

</td>
</tr>
</table>

###### Modify get_seed function ([!7429](https://gitee.com/mindspore/mindspore/pulls/7429/files))

Modify get_seed function implementation

Previously, if seed is not set, the value of seed is default, parameters initialized by the normal function are the same every time.

After modification, if seed is not set, the value of seed is generated randomly, the initialized parameters change according to the random seed.

If you want to fix the initial value of parameters, we suggest to set seed.

```python
>>> from mindspore.common import set_seed
>>> set_seed(1)
```

###### `nn.LinSpace` ([!9494](https://gitee.com/mindspore/mindspore/pulls/9494)) has been removed and modify `ops.LinSpace` ([!8920](https://gitee.com/mindspore/mindspore/pulls/8920))

The `nn.LinSpace` interface only support passing the value by args previously. For the convenience, we provided enhancive `ops.LinSpace` interface, which support passing the value by the inputs at the latest version. So there is no need for `nn.LinSpace`.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore import nn
>>>
>>> start = 1
>>> stop = 10
>>> num = 5
>>> linspace = nn.LinSpace(start, stop, num)
>>> output = linspace()
```

</td>
<td>

```python
>>> import mindspore
>>> from mindspore import Tensor
>>> from mindspore import ops
>>>
>>> linspace = ops.LinSpace()
>>> start = Tensor(1, mindspore.float32)
>>> stop = Tensor(10, mindspore.float32)
>>> num = 5
>>> output = linspace(start, stop, num)
```

</td>
</tr>
</table>

###### Parts of `Optimizer` add target interface ([!6760](https://gitee.com/mindspore/mindspore/pulls/6760/files))

The usage of the sparse optimizer is changed.

The target interface is used to set the execution backend of the sparse operator.

The add_primitive_attr interface is no longer allowed.

The following optimizers add the target interface:  Adam, FTRL, LazyAdam, ProximalAdagrad

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn import Adam
>>>
>>> net = LeNet5()
>>> optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
>>> optimizer.sparse_opt.set_device("CPU")
```

</td>
<td>

```python
>>> from mindspore.nn import Adam
>>>
>>> net = LeNet5()
>>> optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
>>> optimizer.target = 'CPU'
```

</td>
</tr>
</table>

###### `export` Modify the input parameters and export's file name ([!7385](https://gitee.com/mindspore/mindspore/pulls/7385)， [!9057](https://gitee.com/mindspore/mindspore/pulls/9057/files))

Export the MindSpore prediction model to a file in the specified format.

The reference includes: `net`, `*inputs`, `file_name`, `file_format`, `**kwargs`.

Input parameters can be input according to specific export requirements.

Add the file name extension based on the format.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.train.quant import quant
>>>
>>> network = LeNetQuant()
>>> inputs = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
>>> quant.export(network, inputs, file_name="lenet_quant.mindir", file_format='MINDIR')
lenet_quant.mindir
```

</td>
<td>

```python
>>> import mindspore as ms
>>>
>>> network = LeNetQuant()
>>> inputs = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
>>> ms.export(network, inputs, file_name="lenet_quant", file_format='MINDIR', quant_mode='AUTO')
lenet_quant.mindir
```

</td>
</tr>
</table>

###### `Dense`, `Conv2dBnAct`, `DenseBnAct`, `DenseQuant` support setting the activation attribute as an instance of a class derived from `nn.Cell` or `Primtive` ([!7581](https://gitee.com/mindspore/mindspore/pulls/7581))

activation (Union[str, Cell, Primitive]): activate function applied to the output of the fully connected layer

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>>
>>> dense = nn.Dense(1, 1, activation='relu')
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> import mindspore.ops as ops
>>>
>>> dense = nn.Dense(1, 1, activation=nn.ReLU())
>>> dense = nn.Dense(1, 1, activation=ops.ReLU())
```

</td>
</tr>
</table>

###### `tensor.dim()`, `tensor.size()` has been renamed to `tensor.ndim`, `tensor.size` ([!10175](https://gitee.com/mindspore/mindspore/pulls/10175))

Previously, tensor.size() and tensor.dim() were used for checking the total number of elements/dimensions in the tensor.
However, from a user's perspective, tensor.size and tensor.ndim (methods -> properties) are better choices, since they follow the numpy naming convention.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore import Tensor
>>>
>>> Tensor((1,2,3)).size()
>>> Tensor((1,2,3)).dim()
```

</td>
<td>

```python
>>> from mindspore import Tensor
>>>
>>> Tensor((1,2,3)).size
>>> Tensor((1,2,3)).ndim
```

</td>
</tr>
</table>

###### `EmbeddingLookup` add a config in the interface: sparse ([!8202](https://gitee.com/mindspore/mindspore/pulls/8202))

sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: True.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn import EmbeddingLookup
>>>
>>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
>>> result = EmbeddingLookup(4,2)(input_indices)
>>> print(result.shape)
(2, 2, 2)
```

</td>
<td>

```python
>>> from mindspore.nn import EmbeddingLookup
>>>
>>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
>>> result = EmbeddingLookup(4,2)(input_indices, sparse=False)
>>> print(result.shape)
(2, 2, 2)
```

</td>
</tr>
</table>

###### `nn.probability.bijector` change types of attributes from (int, float) to (float, list, numpy.ndarray, Tensor) ([!8191](https://gitee.com/mindspore/mindspore/pulls/8191))

Attributes Type change: (int, float) -> (float, list, numpy.ndarray, Tensor).
Int type is not supported anymore. Parameters of all bijectors should be type float, list, numpy.ndarray or Tensor.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> power = 2
>>> bijector = msb.PowerTransform(power=power)
```

</td>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> power = 2.0
>>> bijector = msb.PowerTransform(power=power)
```

</td>
</tr>
</table>

###### `nn.probability.bijector.GumbelCDF` remove a attribute in the interface: dtype ([!8191](https://gitee.com/mindspore/mindspore/pulls/8191))

dtype is removed from GumbelCDF and is no longer an argument of the class.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>> from mindspore import dtype as mstype
>>>
>>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0, dtype=mstype.float32)
```

</td>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0)
```

</td>
</tr>
</table>

###### `nn.layer.combined.Conv2dBnAct`, `nn.layer.combined.DenseBnAct` move from nn.layer.quant to nn.layer.combined ([!8187](https://gitee.com/mindspore/mindspore/pulls/8187))

Previously Conv2dBnAct and DenseBnAct are in nn.layer.quant, since they are not quant cells, now they are moved to nn.layer.combined. If you import Conv2dBnAct, DenseBnAct from mindspore.nn, then your code doesn't need any change.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn.layer.quant import Conv2dBnAct, DenseBnAct
```

</td>
<td>

```python
>>> from mindspore.nn import Conv2dBnAct, DenseBnAct
```

</td>
</tr>
</table>

###### `nn.layer.conv.Conv2D`, `nn.layer.quant.Conv2dBnFoldQuant`, `nn.layer.quant.Conv2dBnWithoutFoldQuant` change weight shape when group > 1 in Ascend platform ([!9723](https://gitee.com/mindspore/mindspore/pulls/9723))

In Ascend platform, if group > 1, the weight shape of Conv2D change from [in_channels//group, out_channels, kernel_size, kernel_size] to [out_channels, in_channels//group, kernel_size, kernel_size]. Previously, checkpoints of the networks are used, which use Conv2D with group > 1, such as MobileNet, can not be directly used now, need to transpose the first and second axis of the weight.

### Bug fixes

#### FrontEnd

- [STABLE] Fix the problem of the cse optimization in the situation of control flow. (Ascend/GPU)

#### Auto Parallel

- [STABLE] Resolve the restriction: input and output layouts of Reshape are restricted in tensor redistribution. (Ascend/GPU)
- [STABLE] Resolve the restriction: output strategy should be data parallel in model evaluation. (Ascend/GPU)

#### Executor

- [STABLE] Fix fusion operator compilation cache. (Ascend)
- [STABLE] Fix compilation error of dynamic shape operator. (Ascend)
- [STABLE] Fix bug of pynative cannot insert transdata of node output when node should be spilted in the backend opt.(Ascend)
- [STABLE] Fix the bug of TensorMove and memcpy_async merge to one after backend cse pass (Ascend)

#### DataSet

- [STABLE] Fix cache server hang on RequestFreeTag. (Ascend/GPU/CPU)
- [STABLE] Fix hung when use pyfunc multi-processing. (Ascend/GPU/CPU)
- [STABLE] Fix add multiple parent nodes to tree node cause core dump. (Ascend/GPU/CPU)

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Support dynamic shape in MindSpore Lite Converter.
2. Optimize sub-graph mechanism by dynamically splitting the entire graph into multiple subgraphs based on the operator supported, backend hardware and user configuration.
3. Support TensorList and TensorList operators such as TensorListFromTensor, TensorListGetItem and so on.
4. Support BatchMatMul fusion and LSTM fusion in MindSpore Lite Converter.
5. Support converting model and run inference on Windows operator system.
6. Support Model(.ms) visualization on Netron.
7. Support Tensorflow model in MindSpore Lite Converter
8. Add 86 converter parsers.
9. Convert aware training model without user's awareness
10. Support scalar tensor in MindSpore Lite Converter and Runtime
11. Support NPU backend on HUAWEI Kirin SoC.[BETA]
12. Merge timeprofiler into benchmark

#### CPU backend optimization

1. Add 50+ new operators, including new Op type(like Adder, Gru).
2. Enhanced performance on armv8.2 supported platform. For example, utilizing sdot instruction more efficiently.
3. Optimize all operators(fp32, fp16, int8) by implementing multi-thread, SIMD tech as much as possible. Model inference time can reduce at least 20% after these optimizations.
4. Extending to support operators for x86_64 platform based on SSE/AVX instruction set.

#### OpenCL backend

1. Add new ops: add 10+ ops, total 58 ops;
2. Performance optimization: by memory layout optimize, Winograd Convolution select strategyoptimize, SIMT local size optimize, local cache optimize,  GPU performance improvement up to 20+% vs MSLITE Version1.0
3. Add Online Graph optimzation: by fusion Convolution/Matmul/Fullconnection and add/mul/pad/reshape, improve performance up to 50+% for some networks;
4. Add auto tuning: by online tuning in the graph compilation phase, optimize performance up to 10%;
5. Add weight quant: support weight quant
6. Add opencl kernel binary cache: improve Initialization time .

#### Post quantization

MindSpore Lite supports both weight quantization and full quantization. Currently, Weights can be quantized into 1 ~ 16 bits according to user configuration. In internal testing, quantization of networks, such as classification, detection, segmentation and transformer are well supported. To ensure high accuracy of quantized models, MindSpore Lite uses a pipeline quantization method. In the first phase, the weight and activation value are quantized using linear quantization methods, such as MIN-MAX. In the second phase, the quantization error is analyzed, and uses statistical methods to compensate loss caused by fp32 quantization to a fixed point such as Int8 to quantized models. The features of Post-training quantization are:

1. perchannel asymmetric quantization for weights, such as MAX_MIN and KMEANS
2. Perlayer symmetric quantization for activation, such as KL and MAX_MIN.
3. perlayer asymmetrical quantization for activation, such as, RemoveOutlier.
4. accuracy loss compensation, such as BiasCorrection

| mobilenet_v2   | ACC (ImageNet)  |
|---|---|
| FP32  | 71.56%  |
|A8W8   | 71.16%  |
| A8W8(without BiasCorrection)  | 70.74% |
| A8W7  | 71.06%  |
| A7W7  | 70.78%  |

The above table uses the mobilenet_v2 model from TF official website. Using MindSpore Lite quantization, the precision of A8W8 (8-bit activation value quantization and 8-bit weight quantization) decreases from 0.82% to 0.4% after accuracy loss compensation, for 7-bit quantization, the precision loss is still no more than 1%.

#### Training on Device

Within MindSpore 1.1 release, the MindSpore Lite provides the following Training-on-Device (ToD) capabilities:

1. Learning from scratch and Transfer Learning strategies are supported
2. MindSpore based models can be converted and used in training on the device. (Third-party models such as TensorFlow and PyTorch for now cannot be directly imported to the framework)
3. Grad operations are supported for more than 30 operators such as Dense layers, Convolutions and Batch Normalizations. Momentum, SGD, and ADAM optimizers are supported.
4. Supports networks such as LeNet, Alexnet, Resnet, MobileNetV1/V2/V3, and EffectiveNet, and provides complete model loading, conversion, and Python training scripts on the device side.

The MindSpore Lite ToD framework is already in use in the newest Huawei Smart TV, providing a unique and personalized user experience as a family entertainment center.

### API Change

#### API Incompatible Change

##### C++ API

- [Modify] Context now support multi-context configuration.(Context.h)
- [Modify] Callback is move from lite_session.h into ms_tensor.h.
- [Modify] GetInputsByName in lite_session.h is changed into GetInputsByTensorName
- [Add] add static LiteSession *CreateSession(const char*model_buf, size_t size, const lite::Context *context) in lite_session.h
- [Add] add GetErrorInfo interface returning error message in errorcode.h
- [Delete] Remove model_generated.h, ops_generated.h and headers of FlatBuffers library from interfaces

##### Java API

- [Add] Implement JNI layer and add Java api for CPU and GPU backend

#### Deprecations

##### C++ API

Deprecate Interface GetOutputsByNodeName

### Bug fixes

- [BUGFIX] Fix the bug in sub-graph segmentation
- [BUGFIX] Fix the bug in Tensor getitem in which the ellipsis matches the wrong dim-size.
- [BUGFIX] Fix the bug that activation modification after defining Dense will not take effect.

## Contributors

Thanks goes to these wonderful people:

zhouyifengCode, huqi, JulyAi, damon0626, chenbo116, rmdyh, davidmc, gray0v0, doitH, Gogery, zymaa, xinyunfan

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

# MindSpore 1.0.0 Release Notes

## Major Features and Improvements

### MindSpore Training and Inference Framework

#### Ascend

- New models
    - DenseNet121: a dense convolutional neural network, which connects each layer to every other layer in a feed-forward fashion for object recognition on ImageNet dataset.
    - UNet2D-Medical: Unet Medical model for 2D image segmentation, Convolutional Networks for Biomedical Image Segmentation on ISBI Challenge database.
- Frontend and user interface
    - Second-Order Optimization
        - Enable second-order optimization for Bert on Ascend, which can achieve a masked lm accuracy of 71.3% in 800 seconds using 8 Ascend (Bert-Large @MLPerf v0.7 dataset).
    - New GNN model BGCF
        - Bayesian Graph Convolutional Filtering network which naturally incorporate the uncertainty in the user-item interaction graph shows excellent recommendation performance on Amazon-Beauty dataset.
    - Add append interface for SequentialCell.
    - Add a level `auto` for AMP.
- Executor and performance optimization
    - Support quantitative network (Resnet50 & YoloV3 & MobileNetV2).
    - Project ease of use optimization: project compilation time optimization, CMakelist regularization, cudnn, cuda independent compilation and installation independent.
- Data processing, augmentation, and save format
    - Support GeneratorDataset return string type

#### Other Hardware Support

- GPU platform
    - Enable second-order optimization for resnet50 on GPU, which achieve 30% improvement on training time compared to SGD with Momentum (Resnet50 @ImageNet).

#### User interfaces change log

- Remove global object GradOperation in Autodiff([!5011](https://gitee.com/mindspore/mindspore/pulls/5011))
- Remove useless attribute 'name' in Autodiff([!5172](https://gitee.com/mindspore/mindspore/pulls/5172))
- Rectification distributed init([!5350](https://gitee.com/mindspore/mindspore/pulls/5350))
- Move the setting of ParalleMode from train.parallel_utils to context([!5351](https://gitee.com/mindspore/mindspore/pulls/5351))
- Modification of save_checkpoint([!5482](https://gitee.com/mindspore/mindspore/pulls/5482))
- Wrap numpy random seed into an api([!5634](https://gitee.com/mindspore/mindspore/pulls/5634))
- Delete enable_fused_layernorm in some modelzoo scripts([!5665](https://gitee.com/mindspore/mindspore/pulls/5665))
- Move 'multi-subgraphs' interface to internal([!5696](https://gitee.com/mindspore/mindspore/pulls/5696))
- Rename mirror_mean to gradient_mean([!5700](https://gitee.com/mindspore/mindspore/pulls/5700))
- Remove default value of 'group' of DepthWiseConv2d([!5865](https://gitee.com/mindspore/mindspore/pulls/5865))
- Modify interface for function and remove duplicated def([!5958](https://gitee.com/mindspore/mindspore/pulls/5958))
- Unify Conv2d and DepthwiseConv2d([!5916](https://gitee.com/mindspore/mindspore/pulls/5916))
- Modification of SoftmaxCrossEntropyWithLogits([!5502](https://gitee.com/mindspore/mindspore/pulls/5502))
- Change API set_strategy() to shard()([!5991](https://gitee.com/mindspore/mindspore/pulls/5991))
- Move batch_size from bert_cfg_cfg to cfg([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
- Remove unused parameters from SummaryRecord __init__([!5548](https://gitee.com/mindspore/mindspore/pulls/5548))
- remove sens parameter of TrainOneStepWithLossScaleCell([!5753](https://gitee.com/mindspore/mindspore/pulls/5753))
- optimize the TrainOneStepCell for user's define([!6159](https://gitee.com/mindspore/mindspore/pulls/6159))
- delete seed0 and seed1 of nn.Dropout([!5735](https://gitee.com/mindspore/mindspore/pulls/5735))
- delete DataWrapper([!6101](https://gitee.com/mindspore/mindspore/pulls/6101))
- LSTM API optimization([!6374](https://gitee.com/mindspore/mindspore/pulls/6374))
- Merge P\C\F of ops([!5645](https://gitee.com/mindspore/mindspore/pulls/5645))
- delete SoftmaxCrossEntropyExpand interface([!6607](https://gitee.com/mindspore/mindspore/pulls/6607))
- Adjust GroupNorm interface([!6329](https://gitee.com/mindspore/mindspore/pulls/6329))
- Modify init interface to internal interface([!6651](https://gitee.com/mindspore/mindspore/pulls/6651))
- Log optimization([!5842](https://gitee.com/mindspore/mindspore/pulls/5842))
- Remove useless API dataset.set_dataset_size（[!5806](https://gitee.com/mindspore/mindspore/pulls/5806))
- Some of Dataset API add usage parameter（[!5605](https://gitee.com/mindspore/mindspore/pulls/5605))
- Change the import path, such as from mindspore.dataset.transforms.vision to mindspore.dataset.vision.transforms（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Rename ImageFolderDatasetV2 to ImageFolderDataset（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Dataset.map parameter optimization（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Remove useless API MindRecord finish（[!5580](https://gitee.com/mindspore/mindspore/pulls/5580))

### MindSpore Lite

- Converter
    - Add 6 TFLite op, 7 Caffe op, 1 ONNX op.
    - Add support for Windows.
    - Support parallel inference of multiple sessions to adapt to more scenarios
    - Support 8bits only weight-quantization, most main-stream models has small accuracy loss (less than 0.5%) when compared to non-qunantized fp32 model.

- CPU & GPU
    - Add 20 CPU ops，include FP32, int8/uint8, FP16 and int32 ops.
    - Add supporting FP16 for GPU, add 14 GPU ops include FP32/FP16.
    - Add Buffer/Image2D transform op for GPU
    - Performance optimization for CPU ops focus on ARM32.
    - Performance optimization for GPU Convolution using winograd.

- Tool & example
    - Add object detection Android Demo.

## Bugfixes

- Models
    - fix the constant folding problem in multiply.([!6092](https://gitee.com/mindspore/mindspore/pulls/6092))
    - move batch_size from bert_net_cfg to cfg in bert scripts.([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
    - modify the checkpoint file path.([!6137](https://gitee.com/mindspore/mindspore/pulls/6137))
- Python API
    - fix semi auto parallel parameter of reshape has another user([!5722](https://gitee.com/mindspore/mindspore/pulls/5722))
    - raise ValueError when call hook function in graph mode([!5831](https://gitee.com/mindspore/mindspore/pulls/5831))
- Executor
    - fix pynative mode to build temporary nn objects.（[!6189](https://gitee.com/mindspore/mindspore/pulls/6189))
    - fix the accuracy problem of multiple inputs of multi-card communication operator broadcast.([!6522](https://gitee.com/mindspore/mindspore/pulls/5622))
    - fix the problem that the sample distribution interface categorical does not support graph mode.([!5772](https://gitee.com/mindspore/mindspore/pulls/5772))
    - fix the random seed failure problem of the polynomial downsampling distribution operator.([!5948](https://gitee.com/mindspore/mindspore/pulls/5948))
    - fix unnecessary address binding issues in GPU heterogeneous scenarios.([!6232](https://gitee.com/mindspore/mindspore/pulls/6232))
- GPU platform
    - fix for kernel resource leak([!5315](https://gitee.com/mindspore/mindspore/pulls/5315))
    - fix for insufficient memory for continuous unit test running([!5617](https://gitee.com/mindspore/mindspore/pulls/5617))
    - fix for the memory leak in the sparse slicer([!5578](https://gitee.com/mindspore/mindspore/pulls/5578))
- Data processing
    - fix hang when use pyfunc([!6346](https://gitee.com/mindspore/mindspore/pulls/6346))
    - fix GPU device queue does not release GIL during resource clean up([!5964](https://gitee.com/mindspore/mindspore/pulls/5964))
    - fix hang if scripte exit unnormally([!6441](https://gitee.com/mindspore/mindspore/pulls/6441))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, danish, Danish, dayschan, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, gongdaguo, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, root, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, Zichun, Zirui, Ziyan, zjun, ZPaC

Contributions of any kind are welcome!

# MindSpore 0.7.0-beta Release Notes

## Major Features and Improvements

### MindSpore Training and Inference Framework

#### Ascend

- New models
    - TinyBert: a smaller and faster version of BERT using transformer distillation for natural language understanding on GLUE benchmark.
    - SE-ResNet50: add Squeeze-and-Excitation blocks(SE-Blocks) to the resnet50 network to improve channel interdependencies for image classification on ImageNet 2012 dataset.
    - Inception V3: the third version of Inception convolutional architectures for image classification on ImageNet 2012 dataset.
- Frontend and user interface
    - Embedding operator high-level packaging to support segmented by field for Wide&Deep.
    - Load multi-node checkpoint into single-process to support host-device hybrid inference.
    - Support Concat/Tile/Strideslice distributed operators.
    - Support cumulative gradient and batch training split.
    - Support variable parameter input for Cell object.
    - Parameter mixed calculation optimization for pynative mode.
    - Deep Probabilistic Programming
        - Support statistical distributions classes used to generate stochastic tensors.
        - Support probabilistic inference algorithms.
        - Support BNN layers used to construct BNN in Graph mode.
        - Support interfaces for the transformation between BNN and DNN in Graph mode.
        - Support uncertainty estimation to estimate epistemic uncertainty and aleatoric uncertainty.
    - User interfaces change log
        - change base class of parameter([!3473](https://gitee.com/mindspore/mindspore/pulls/3473))
        - change binary to mindir([!4258](https://gitee.com/mindspore/mindspore/pulls/4258))
        - change export from geir to air([!4269](https://gitee.com/mindspore/mindspore/pulls/4269))
        - Init parameter data by default([!3967](https://gitee.com/mindspore/mindspore/pulls/3967))
        - change IndexedSlices to RowTensor([!4031](https://gitee.com/mindspore/mindspore/pulls/4031))
        - Must set or change parallel mode before any Initializer created([!4801](https://gitee.com/mindspore/mindspore/pulls/4801))
- Executor and performance optimization
    - MindSpore graph compilation process performance improved by 20%.
    - Decoupling C++ and Python modules to achieve separate compilation of core modules.
- Data processing, augmentation, and save format
    - Support automatic data augmentation
    - Support GNN distributed cache in single node
    - Support ConcatDataset using distributed sampler

#### Other Hardware Support

- GPU platform
    - New model supported: VGG16, ResNet101, DeepFM.
    - Support some distributed operators in ResNet50 and Wide&Deep.
    - Support automatic parallel for Wide&Deep.
    - Support function funcs[i](*inputs) (such as switch-case).
    - Support distributed training with parameter server.
    - Support GPU operator profiling.
    - Performance optimization of the distributed training with allreduce.
    - Performance optimization of the mixed precision training.
    - Performance optimization of the pynative mode.
    - Performance optimization of the convolution operator, batch normalization operator.
- CPU platform
    - Support MobileNetV2 Re-Training: Re-train the network with different class number.

### MindSpore Lite

- Converter
    - Support third-party models, including TFLite/Caffe/ONNX.
    - Add 93 TFLite op.
    - Add 24 Caffe op.
    - Add 62 ONNX op.
    - Add 11 optimized passes, include fusion/const fold.
    - Support aware-training and Post-training quantization.
- CPU
    - Add 100+ops，support fp32, int8/uint8, FP16 ops
    - Support fast convolution algorithms: Sliding Window, Img2col + Gemm, Strassen, Winograd
    - Support assembly/neon instruction.
    - Support CPU fp16 and sdot on ARM v8.2+.
- GPU
    - Add 20+ ops for OpenCL.
    - Support image2D/buffer format.
    - Optimize online initialization time.
    - add optimized convolution1X1/3X3/depthwise/convolution_transposed for OpenCL.
- Tool & example
    - Add benchmark and TimeProfile tools.
    - Add image classification Android Demo.

## Bugfixes

- Models
    - normalize the readme file([!5410](https://gitee.com/mindspore/mindspore/pulls/5410))
    - fix a sink_size bug for transformer([!5393](https://gitee.com/mindspore/mindspore/pulls/5393))
    - fix bool type optional for resnet50([!5363](https://gitee.com/mindspore/mindspore/pulls/5363))
- Python API
    - improve interface '__bool__' for tensor([!4000](https://gitee.com/mindspore/mindspore/pulls/4000))
    - fix GPU-ResizeNearestNeighbor([!3760](https://gitee.com/mindspore/mindspore/pulls/3760))
    - fix topK multi dimension grad func([!3711](https://gitee.com/mindspore/mindspore/pulls/3711))
    - fix scatterop error msg([!3699](https://gitee.com/mindspore/mindspore/pulls/3699))
    - fix bug of cast dtype when using mix_presion in pynative mode([!3730](https://gitee.com/mindspore/mindspore/pulls/3730))
- Executor
    - fix etsnet train error when UnsegmentSum's first input shape is (1,) ([!4573](https://gitee.com/mindspore/mindspore/pulls/4573))
    - fix bug of result error in while control flow because of unsupporting for value reference ([!4103](https://gitee.com/mindspore/mindspore/pulls/4103))
    - fix bug of the output tensor does not carry device data type ([!3774](https://gitee.com/mindspore/mindspore/pulls/3774))
    - fix bug of avoiding multi attr value are eliminated in pynative mode ([!4225](https://gitee.com/mindspore/mindspore/pulls/4225))
    - fix bug of AssignAdd unable to work normally in multi-cases ([!5171](https://gitee.com/mindspore/mindspore/pulls/5171))
- GPU platform
    - improve the environment variable checking for nvcc compiler path ([!5140](https://gitee.com/mindspore/mindspore/pulls/5140))
    - fix bug of error in cast operator conversion from fp16 to fp32 ([!4147](https://gitee.com/mindspore/mindspore/pulls/4147))
    - fix bug of the array out of bound in case of make_tuple operator ([!5219](https://gitee.com/mindspore/mindspore/pulls/5219))
- Data processing and Pro
    - fix GeneratorDataset time out([!3624](https://gitee.com/mindspore/mindspore/pulls/3624))
    - fix concat operator get_dataset_size error([!4701](https://gitee.com/mindspore/mindspore/pulls/4701))
    - fixing python validator for Repeat Op([!4366](https://gitee.com/mindspore/mindspore/pulls/4366))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, Alexey, andy, andy_wangrui, anthonyaje, anzhengqi, askmiao, avakh, baihuawei, bingyaweng, BowenK, buxue, caifubi, CaoJian, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chenzupeng, chujinjin, cjh9368, Corleone, cristoval, danish, dengyutao, eric, Eric, ervinzhang, etone-chan, fangzehua, fary86, fuzhiye, gengdongjie, genglishuai, Giancarlo, gongdaguo, gukecai, guohongzilong, GuoMengHao, hangq, hanhaocheng, hanhuifeng2020, hanjun996, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, hongxing, huangdongrun, huanghui, huangxinjing, islam_amin, Jesse, jianghui58, jiangzhiwen, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, kai00, kingfo, kpy, kswang, laiyongqiang, leilei_snow, leopz, Li, liangzelang, lianliguang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, lingyunli63, linqingke, lirongzhen1, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuzhongkai, Lixia, lixian, liyong, lizhenyu, looop5, luoyang, lvchangquan, lvliang, lvwenyuan, lyvette, mahdi, Mahdi, mamba_ni, maning202007, Margaret_wangrui, mayang, meixiaowei, meng_chunyang, ms_yan, nhussain, panbingao, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, pengyongrong, Pengyongrong, qianlong, qujianwei, root, shenwei41, shibeiji, simson, songhonglei413, Su, sunsuodong, suteng, tao_yunhao, TFbunny, tinazhang, tom__chen, tony_liu2, tronzhang, VectorSL, wandongdong, wangdongxu, wanghua, wangmin, wangshaocong, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, wuyongkang, xiefangqi, xuanyue, Xun, xutianchun, xuyongfei, yanghaitao, yangjie159, YangLuo, yangruoqi713, yangyongjie, yangzhenzhang, yankai, yao_yf, yelihua, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zhangxuetong, zhaizhiqiang, Zhang, zhangxinfeng3, zhangxuetong, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaoting, zhaozhenlong, zhengjun10, zhongligeng, zhoufeng, zhousiyi, zhouyaqiang, zhouyuanshen, Zichun, Zirui, zjun, zongha, ZPaC, lijiaqi, liangchenghui, wangminggui

Contributions of any kind are welcome!

# MindSpore 0.6.0-beta Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - There are official, research and community under modelzoo.
        - Official is maintained  with the newest APIs by MindSpore team,  MaskRCNN are added.
        - Research is uploaded by researchers for official review, and APIs may not  be updated in time.
        - Community reprints the relevant links of partner research results.
    - Hub added on the same level as modelzoo, synchronous storage of materials needed for official hub web pages which will be launched soon.
    - Support pre-trained models, few lines of code can be used to download and load pre-trained models, supporting inference or transfer learning.
- Frontend and user interface
    - Supports user side operator compilation and graph execution error rendering.
    - Uniform definition dynamic learning rate behavior in optimizers.
    - Support IndexSlice in sparse expression.
    - Support use parent construct method during construct.
    - Support asynchronous execution save checkpoint file.
    - Support implicit type conversion in pynative mode.
    - User interfaces change log
        - unform learning rate behavior in optimizers([!2755](https://gitee.com/mindspore/mindspore/pulls/2755))
        - rename operator of sparse optimizer([!3217](https://gitee.com/mindspore/mindspore/pulls/3217))
        - move profiler module from mindinsight to mindspore([!3075](https://gitee.com/mindspore/mindspore/pulls/3075))
        - VOCDataset output change to multi-columns([!3093](https://gitee.com/mindspore/mindspore/pulls/3093))
        - GetDatasize feature([!3212](https://gitee.com/mindspore/mindspore/pulls/3212))
        - dataset: modify config api([!2936](https://gitee.com/mindspore/mindspore/pulls/2936))
- Executor and performance optimization
    - Decouple C++ and python, so make the architecture more extensible.
    - Parameter Server for distributed deep learning supported.
    - Serving: a flexible service deployment framework for deep learning models.
    - Memory reuse is enhanced, and the batch size of Bert large model is increased from 96 to 160 on a single server.
- Data processing, augmentation, and save format
    - Support MindRecord save operator after  date processing
    - Support automatic fusion operator, such as decode/resize/crop
    - Support CSV dataset loading

### Other Hardware Support

- GPU platform
    - New model supported: ResNext50, WarpCTC and GoogLeNet.
    - Support hyperparametric search and data enhanced automl on GPU.
    - Support Resnet50 automatic parallel in GPU backend.

## Bugfixes

- Models
    - Improved the performance and accuracy on ResNet50([!3456](https://gitee.com/mindspore/mindspore/pulls/3456))
    - Fixed the performance test case of bert([!3486](https://gitee.com/mindspore/mindspore/pulls/3486))
- Python API
    - Fix assign used in while loop([!2720](https://gitee.com/mindspore/mindspore/pulls/2720))
    - Revert optimize the graph output of all nop node.([!2857](https://gitee.com/mindspore/mindspore/pulls/2857))
    - Print tensor as numpy.([!2859](https://gitee.com/mindspore/mindspore/pulls/2859))
    - Support weight decay for sparse optimizer([!2668](https://gitee.com/mindspore/mindspore/pulls/2668))
    - Fix BatchToSpaceND([!2741](https://gitee.com/mindspore/mindspore/pulls/2741))
    - Fixing type check mistakes of InplaceAdd and Inplace Sub ops([!2744](https://gitee.com/mindspore/mindspore/pulls/2744]))
    - Change order param only equal to group param([!2748](https://gitee.com/mindspore/mindspore/pulls/2748))
- Executor
    - The performance of graph with control flow is optimized([!2931](https://gitee.com/mindspore/mindspore/pulls/2931))
    - Fix bug of wrong number of tuple layers([!3390](https://gitee.com/mindspore/mindspore/pulls/3390))
    - Fix cpu multi graph memory exception([!3631](https://gitee.com/mindspore/mindspore/pulls/3631))
    - Enable data sync when calling operator without defining a cell([!3081](https://gitee.com/mindspore/mindspore/pulls/3081))
    - Fix argmaxwith value error in pynative mode on GPU([!3082](https://gitee.com/mindspore/mindspore/pulls/3082))
    - Fix precision error with fp16 input on pynative mode([!3196](https://gitee.com/mindspore/mindspore/pulls/3196))
- Data processing
    - Fix bug of RandomColor and RandomSharpness default parameter checking  ([!2833](https://gitee.com/mindspore/mindspore/pulls/2833))
    - Fix process hung when training and eval  ([!3469](https://gitee.com/mindspore/mindspore/pulls/3469))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.5.2-beta Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - DenseNet121: a convolution based neural network for the task of image classification on ImageNet 2012 dataset.

## Bugfixes

- Models
    - VGG16,Alexnet,GoogleNet,optimize network for better performance. ([!5539](https://gitee.com/mindspore/mindspore/pulls/5539))
    - YOLOV3, fix yolov3_darknet53 dataset bug. ([!5658](https://gitee.com/mindspore/mindspore/pulls/5658))

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.5.0-beta Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - ResNext50: a simple, highly modularized network architecture using aggregated resdiual transformations for image classification on ImageNet 2012 dataset.
    - MASS: a pre-training method for sequence to sequence based language generation tasks on Text Summarization and Conversational Response Generation using News Crawls 2007-2017 dataset, Gigaword corpus and Cornell movie dialog corpus.
    - Transformer: a neural network architecture for language understanding on WMT 2014 English-German dataset.
    - GCN: Graph Convolutional Networks for the task of classification of nodes in a graph on Cora and Citeseer datasets.
    - GAT: an attention-based graph neural network for node classification on Cora and CiteSeer dataset.
- Frontend and user interface
    - Support tensor value and assignment of mixed tensor index in graph mode.
    - Support tensor comparison, len operator, constexpr syntax, value and assignment of tensor index in pynative mode.
    - Support converting MindSpore IR to pb format for infer model.
    - Support print operator to write data directly on the hard disk.
    - Add the double recursive programming solution for very high speed parallel strategy search in automatic parallel.
    - User interfaces change log
        - Allow the learning rate of AdamWeightDecayDynamicLR and Lamb to be 0([!1826](https://gitee.com/mindspore/mindspore/pulls/1826))
        - Restricting the entire network input parameter is Tensor([!1967](https://gitee.com/mindspore/mindspore/pulls/1967))
        - Turn shape and dtype into attributes instead of interfaces([!1919](https://gitee.com/mindspore/mindspore/pulls/1919))
        - Delete multitypefungraph([!2116](https://gitee.com/mindspore/mindspore/pulls/2116))
        - Refactor the callback module in an encapsulated way, use _CallbackManager instead of_build_callbacks([!2236](https://gitee.com/mindspore/mindspore/pulls/2236))
        - Delete EmbeddingLookup([!2163](https://gitee.com/mindspore/mindspore/pulls/2163))
        - Checkpoint add model_type([!2517](https://gitee.com/mindspore/mindspore/pulls/2517))
- Executor and performance optimization
    - Heterogeneous execution on CPU and Ascend devices supported, and is verified in Wide&Deep model.
    - Quantitative training of MobileNetV2, Lenet and Resnet50 on Ascend are supported.
    - Support new fusion architecture, which can do fusion optimization across graphs and kernels to improve execution speed.
- Data processing, augmentation, and save format
    - Support data processing pipeline performance profiling.
    - Support public dataset loading, such as CLUE and Coco.
    - Support more text processing, such as more tokenizers and vocab data.
    - Support MindRecord padded data.

### Other Hardware Support

- GPU platform
    - New model supported: Bert / Wide&Deep.
    - Support setting max device memory.
- CPU platform
    - New model supported: LSTM.

## Bugfixes

- Models
    - Bert, Move Bert from `example` to `model_zoo`, optimize network for better performance. ([!1902](https://gitee.com/mindspore/mindspore/pulls/1902))
    - VGG16, Move VGG16 from `example` to `model_zoo`, optimize network for better accuracy. ([!2645](https://gitee.com/mindspore/mindspore/pulls/2645))
    - Alexnet, modify parameter setting to improve accuracy ([!1364](https://gitee.com/mindspore/mindspore/pulls/2370))
    - Wide&Deep, Move Wide&Deep from `example` to `model_zoo`, optimize network for better performance. ([!2221](https://gitee.com/mindspore/mindspore/pulls/2221))
- Python API
    - Fix bug in auto cast([!1766](https://gitee.com/mindspore/mindspore/pulls/1766))
    - Fix bug of register_backward_hook([!2148](https://gitee.com/mindspore/mindspore/pulls/2148))
    - Fix bug of tuple args in pynative mode([!1878](https://gitee.com/mindspore/mindspore/pulls/1878))
    - Fix bug of checking numbers of arguments and graph parameters([!1701](https://gitee.com/mindspore/mindspore/pulls/1701))
- Executor
    - Fix bug of loading input data repeatedly in pynative mode([!1966](https://gitee.com/mindspore/mindspore/pulls/1966))
    - Fix bug of list cannot be used as input in pynative mode([!1765](https://gitee.com/mindspore/mindspore/pulls/1765))
    - Fix bug of kernel select ([!2103](https://gitee.com/mindspore/mindspore/pulls/2103))
    - Fix bug of pattern matching for batchnorm fusion in the case of auto mix precision.([!1851](https://gitee.com/mindspore/mindspore/pulls/1851))
    - Fix bug of generate hccl's kernel info.([!2393](https://gitee.com/mindspore/mindspore/pulls/2393))
- GPU platform
    - Fix bug of summary feature invalid([!2173](https://gitee.com/mindspore/mindspore/pulls/2173))
- Data processing
    - Fix bug of Cifar dataset reading([!2096](https://gitee.com/mindspore/mindspore/pulls/2096))
    - Fix bug of C++ behavior in RandomCropAndResize([!2026](https://gitee.com/mindspore/mindspore/pulls/2026))
    - Fix the bug of mindrecord shuffle([!2420](https://gitee.com/mindspore/mindspore/pulls/2420))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.3.1-alpha Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- Frontend and User Interface
    - Independent model init interface.
- Data processing, augmentation, and save format
    - Support sample padding for minddataset.

## Bugfixes

- Python API
    - Fix bugs in the lars optimizer([!1894](https://gitee.com/mindspore/mindspore/pulls/1894))
- Data processing
    - Fix accuracy problem of RandomCropDecodeResize ([!2340](https://gitee.com/mindspore/mindspore/pulls/2340))

# Release 0.3.0-alpha

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - DeepFM: a factorization-machine based neural network for CTR prediction on Criteo dataset.
    - DeepLabV3: significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2007 semantic image segmentation benchmark.
    - Faster-RCNN: towards real-time object detection with region proposal networks on COCO 2017 dataset.
    - SSD: a single stage object detection methods on COCO 2017 dataset.
    - GoogLeNet: a deep convolutional neural network architecture codenamed Inception V1 for classification and detection on CIFAR-10 dataset.
    - Wide&Deep: jointly trained wide linear models and deep neural networks for recommender systems on Criteo dataset.
- Frontend and User Interface
    - Complete numpy advanced indexing method. Supports value and assignment through tensor index.
    - Some optimizers support separating parameter groups. Different parameter groups can set different `learning_rate` and `weight_decay`.
    - Support setting submodule's logging level independently, e.g. you can set logging level of module `A` to warning and set logging level of module `B` to info.
    - Support weights to be compiled according to shape to solve the problem of large memory overhead.
    - Add some operators implement and grammar support in pynative mode. To be consistent with graph mode.
    - User interfaces change log
        - Learning rate and weight decay making group params([!637](https://gitee.com/mindspore/mindspore/pulls/637))
        - Support weights to be compiled according to shape([!1015](https://gitee.com/mindspore/mindspore/pulls/1015))
        - delete some context param([!1100](https://gitee.com/mindspore/mindspore/pulls/1100))
        - ImageSummary/ScalarSummary/TensorSummary/HistogramSummary([!1329](https://gitee.com/mindspore/mindspore/pulls/1329))([!1425](https://gitee.com/mindspore/mindspore/pulls/1425))
- Executor and Performance Optimization
    - Support doing evaluation while in training process, so that the accuracy of training can be easily obtained.
    - Enable second-order optimization for resnet50, which can achieve 75.9% accuracy in 45 epochs (Resnet50 @ImageNet).
    - Optimize pynative implementation and improve it's execution performance.
    - Optimize summary record implementation and improve its performance.
- Data processing, augmentation, and save format
    - Support simple text processing, such as tokenizer/buildvocab/lookup.
    - Support padding batch.
    - Support split or concat dataset.
    - Support MindDataset reading from file list.

### Other Hardware Support

- GPU platform
    - New models supported: MobileNetV2, MobileNetV3.
    - Support mixed precision training.
    - Support device memory swapping.

## Bugfixes

- Python API
    - An exception to the broadcast input data type check([!712](https://gitee.com/mindspore/mindspore/pulls/712))
    - Fix issues assignsub return value 0([!1036](https://gitee.com/mindspore/mindspore/pulls/1036))
    - Fix issue Conv2dBackpropInput bprop should return 3 instead of 2 items([!1001](https://gitee.com/mindspore/mindspore/pulls/1001))
    - Fix sens shape error of TrainOneStepWithLossScaleCell([!1050](https://gitee.com/mindspore/mindspore/pulls/1050))
    - Fix BatchNormGrad operator([!1344](https://gitee.com/mindspore/mindspore/pulls/1344))
- Executor
    - Fix dropout，topK and addn errors in PyNative mode ([!1285](https://gitee.com/mindspore/mindspore/pulls/1285), [!1138](https://gitee.com/mindspore/mindspore/pulls/1138), [!1033](https://gitee.com/mindspore/mindspore/pulls/1033)).
    - Fix memory leaks after execution in PyNatvie mode ([!1201](https://gitee.com/mindspore/mindspore/pulls/1201)).
    - Fix HCCL failure in some special scenes ([!1204](https://gitee.com/mindspore/mindspore/pulls/1204), [!1252](https://gitee.com/mindspore/mindspore/pulls/1252)).
    - Fix SSD network when Select failed, can't find kernel info([!1449](https://gitee.com/mindspore/mindspore/pulls/1449)).
    - Fix Topk operator selection strategy bug between aicore and aicpu([!1367](https://gitee.com/mindspore/mindspore/pulls/1367)).
    - Fix input memory size of 'assign' op unequal in control sink mode when assigning a data from one child graph to another child graph([!802](https://gitee.com/mindspore/mindspore/pulls/802)).
    - Fix allreduce ir inconsistency([!989](https://gitee.com/mindspore/mindspore/pulls/989)).
- GPU platform
    - Fix summary for gradient collection ([!1364](https://gitee.com/mindspore/mindspore/pulls/1364))
    - Fix the slice operator ([!1489](https://gitee.com/mindspore/mindspore/pulls/1489))
- Data processing
    - Fix memory problems of GeneratorDataset of sub-process ([!907](https://gitee.com/mindspore/mindspore/pulls/907))
    - Fix getting data timeout when training the cifar10 dataset under the lenet([!1391](https://gitee.com/mindspore/mindspore/pulls/1391))

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, Amir Lashkari, anthony, baihuawei, biffex, buxue, caifubi, candanzg, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenzomi, chujinjin, cristoval, dengwentao, eric, etone-chan, fary86, gaojing, gengdongjie, gongchen, guohongzilong, guozhijian, heleiwang, hesham, He Wei, Hoai Linh Tran, hongxing, huangdongrun, huanghui, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jonwe, jonyguo, Junhan Hu, Kang, kingfo, kswang, laiyongqiang, leopz, lichenever, lihongkang, limingqi107, liubuyu, liuliyan2, liuwenhao4, liuxiao, liuxiao, liyong, lizhenyu, lvliang, Margaret_wangrui, meixiaowei, ms_yan, Nat Sutyanyong, ougongchang, panfengfeng, panyifeng, Peilin Wang, peixu_ren, qianlong, rick_sanchez, seatea, sheng, shijianning, simson, sunsuodong, Tinazhang, VectorSL, wandongdong, wangcong, wanghua, wangnan39, Wei Luning, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuxuejian, Xiaoda Zhang, xiefangqi, xulei2020, Yang, yangjie159, yangruoqi713, yangyongjie, yangzhenzhang, Yanjun Peng, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yujianfeng, YuJianfeng, yvetteliu, zhangdengcheng, Zhang Qinghua, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, zhouyuanshen, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang

Contributions of any kind are welcome!

# MindSpore 0.2.0-alpha Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    - ResNet101: Deep Residual Learning for Image Recognition.

- Frontend and User Interface
    - Support for all python comparison operators.
    - Support for math operators **,//,%. Support for other python operators like and/or/not/is/is not/ in/ not in.
    - Support for the gradients of function with variable arguments.
    - Support for tensor indexing assignment for certain indexing type.
    - Support for dynamic learning rate.
    - User interfaces change log
        - DepthwiseConv2dNative, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropInput([!424](https://gitee.com/mindspore/mindspore/pulls/424))
        - ReLU6, ReLU6Grad([!224](https://gitee.com/mindspore/mindspore/pulls/224))
        - GeneratorDataset([!183](https://gitee.com/mindspore/mindspore/pulls/183))
        - VOCDataset([!477](https://gitee.com/mindspore/mindspore/pulls/477))
        - MindDataset, PKSampler([!514](https://gitee.com/mindspore/mindspore/pulls/514))
        - map([!506](https://gitee.com/mindspore/mindspore/pulls/506))
        - Conv([!226](https://gitee.com/mindspore/mindspore/pulls/226))
        - Adam([!253](https://gitee.com/mindspore/mindspore/pulls/253))
        - _set_fusion_strategy_by_idx,_set_fusion_strategy_by_size([!189](https://gitee.com/mindspore/mindspore/pulls/189))
        - CheckpointConfig([!122](https://gitee.com/mindspore/mindspore/pulls/122))
        - Constant([!54](https://gitee.com/mindspore/mindspore/pulls/54))
- Executor and Performance Optimization
    - Support parallel execution of data prefetching and forward/backward computing.
    - Support parallel execution of gradient aggregation and forward/backward computing in distributed training scenarios.
    - Support operator fusion optimization.
    - Optimize compilation process and improve the performance.
- Data processing, augmentation, and save format
    - Support multi-process of GeneratorDataset/PyFunc for high performance
    - Support variable batchsize
    - Support new Dataset operators, such as filter,skip,take,TextLineDataset

### Other Hardware Support

- GPU platform
    - Use dynamic memory pool by default on GPU.
    - Support parallel execution of computation and communication.
    - Support continuous address allocation by memory pool.
- CPU platform
    - Support for windows 10 OS.

## Bugfixes

- Models
    - Fix mixed precision bug for VGG16 model ([!629](https://gitee.com/mindspore/mindspore/pulls/629)).
- Python API
    - Fix ControlDepend operator bugs on CPU and GPU ([!396](https://gitee.com/mindspore/mindspore/pulls/396)).
    - Fix ArgMinWithValue operator bugs ([!338](https://gitee.com/mindspore/mindspore/pulls/338)).
    - Fix Dense operator bugs on PyNative mode ([!276](https://gitee.com/mindspore/mindspore/pulls/276)).
    - Fix MatMul operator bugs on PyNative mode ([!288](https://gitee.com/mindspore/mindspore/pulls/288)).
- Executor
    - Fix operator selection bugs and make it general ([!300](https://gitee.com/mindspore/mindspore/pulls/300)).
    - Fix memory reuse bug for GetNext op ([!291](https://gitee.com/mindspore/mindspore/pulls/291)).
- GPU platform
    - Fix memory allocation in multi-graph scenarios ([!444](https://gitee.com/mindspore/mindspore/pulls/444)).
    - Fix bias_add_grad under fp16 precision ([!598](https://gitee.com/mindspore/mindspore/pulls/598)).
    - Fix support for fp16 kernels on nvidia 1080Ti([!571](https://gitee.com/mindspore/mindspore/pulls/571)).
    - Fix parsing of tuple type parameters ([!316](https://gitee.com/mindspore/mindspore/pulls/316)).
- Data processing
    - Fix TypeErrors about can't pickle mindspore._c_dataengine.DEPipeline objects([!434](https://gitee.com/mindspore/mindspore/pulls/434)).
    - Add TFRecord file verification([!406](https://gitee.com/mindspore/mindspore/pulls/406)).

## Contributors

Thanks goes to these wonderful people:

Alexey_Shevlyakov, Cathy, Chong, Hoai, Jonathan, Junhan, JunhanHu, Peilin, SanjayChan, StrawNoBerry, VectorSL, Wei, WeibiaoYu, Xiaoda, Yanjun, YuJianfeng, ZPaC, Zhang, ZhangQinghua, ZiruiWu, amongo, anthonyaje, anzhengqi, biffex, caifubi, candanzg, caojian05, casgj, cathwong, ch-l, chang, changzherui, chenfei, chengang, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, dengwentao, dinghao, fanglei, fary86, flywind, gaojing, geekun, gengdongjie, ghzl, gong, gongchen, gukecai, guohongzilong, guozhijian, gziyan, h.farahat, hesham, huangdongrun, huanghui, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, jonathan_yan, jonyguo, jzw, kingfo, kisnwang, laiyongqiang, leonwanghui, lianliguang, lichen, lichenever, limingqi107, liubuyu, liuxiao, liyong, liyong126, lizhenyu, lupengcheng, lvliang, maoweiyong, ms_yan, mxm, ougongchang, panfengfeng, panyifeng, pengyanjun, penn, qianlong, seatea, simson, suteng, thlinh, vlne-v1, wangchengke, wanghua, wangnan39, wangqiuliang, wenchunjiang, wenkai, wukesong, xiefangqi, xulei, yanghaitao, yanghaoran, yangjie159, yangzhenzhang, yankai10, yanzhenxiang2020, yao_yf, yoonlee666, zhangbuxue, zhangz0911gm, zhangzheng, zhaojichen, zhaoting, zhaozhenlong, zhongligeng, zhoufeng, zhousiyi, zjun, zyli2020, yuhuijun, limingqi107, lizhenyu, chenweifeng.

Contributions of any kind are welcome!

# MindSpore 0.1.0-alpha Release Notes

## Main Features

### Ascend Training and Inference Framework

- Recommended OS: Ubuntu 16.04 (or later) or EulerOS 2.5 or EulerOS 2.8
- Python version: 3.7.5
- Preset models
    - ResNet-50: residual structure-based convolutional neural network (CNN) for image classification, which is widely used.
    - AlexNet: classic CNN for image classification, achieving historical results in ImageNet LSVRC-2012.
    - LeNet: classic CNN for image classification, which was proposed by Yann LeCun.
    - VGG16: classic CNN for image classification, which was proposed by Oxford Visual Geometry Group.
    - YoloV3: real-time object detection network.
    - NEZHA: BERT-based Chinese pre-training network produced by Huawei Noah's Ark Laboratory.
- Execution modes
    - Graph mode: provides graph optimization methods such as memory overcommitment, IR fusion, and buffer fusion to achieve optimal execution performance.
    - PyNative mode: single-step execution mode, facilitating process debugging.
- Debugging capability and methods
    - Save CheckPoints and Summary data during training.
    - Support asynchronous printing.
    - Dump the computing data.
    - Support profiling analysis of the execution process performance.
- Distributed execution
    - Support AllReduce, AllGather, and BroadCast collective communication.
    - AllReduce data parallel: Each device obtains different training data, which accelerates the overall training process.
    - Collective communication-based layerwise parallel: Models are divided and allocated to different devices to solve the problem of insufficient memory for large model processing and improve the training speed.
    - Automatic parallel mode: The better data and model parallel mode can be predicted based on the cost model. It is recommended that this mode be used on ResNet series networks.
- Automatic differentiation
    - Implement automatic differentiation based on Source to Source.
    - Support distributed scenarios and automatic insertion of reverse communication operators.
- Data processing, augmentation, and save format
    - Load common datasets such as ImageNet, MNIST, CIFAR-10, and CIFAR-100.
    - Support common data loading pipeline operations, such as shuffle, repeat, batch, map, and sampler.
    - Provide basic operator libraries to cover common CV scenarios.
    - Support users to customize Python data augmentation operators through the Pyfunc mechanism.
    - Support the access of user-defined datasets through the GeneratorDataset mechanism.
    - Provide the MindSpore data format, data aggregation and storage, random access example, data partition, efficient parallel read, user-defined index, and dataset search.
    - Convert user datasets to the MindSpore data format.
    - After data processing and augmentation, provide training applications in feed and graph modes.
- FP32/16 mixed precision computation, supporting automatic and manual configuration
- Provide common operators such as nn, math, and array, which can be customized.

### Inference Deployment

- Deploy models in MindSpore format on the Ascend 310 platform for inference.
- Save models in ONNX format.
- Support saving models in LITE format and running models based on the lightweight inference framework.
    - Recommended OS: Android 4.3 or later
    - Supported network type: LeNet
    - Provide the generalization operators generated by TVM and operators generated after specific networks are tuned.

### Other Hardware Support

- GPU platform training
    - Recommended OS: Ubuntu 16.04
    - CUDA version: 9.2 or 10.1
    - CuDNN version: 7.6 or later
    - Python version: 3.7.5
    - NCCL version: 2.4.8-1
    - OpenMPI version: 3.1.5
    - Supported models: AlexNet, LeNet, and LSTM
    - Supported datasets: MNIST and CIFAR-10
    - Support data parallel.
- CPU platform training
    - Recommended OS: Ubuntu 16.04
    - Python version: 3.7.5
    - Supported model: LeNet
    - Supported dataset: MNIST
    - Provide only the stand-alone operation version.

## Peripherals and Tools

- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindInsight Visualization Debugging and Optimization](https://gitee.com/mindspore/mindinsight)
- [MindArmour Model Security Hardening Package](https://gitee.com/mindspore/mindarmour)
