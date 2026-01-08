# MindSpore Release Notes

[View English](./RELEASE.md)

## MindSpore 2.7.2 Release Notes

### 主要特性及增强

#### Ascend

- [BETA] 静态图jit_level为O1模式下支持GroupedMatMul + Reshape + AssignAdd融合模式，用户可通过设置环境变量 `export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_cluster_ops=GroupedMatmul,Reshape"` 开启。
- [STABLE] 静态图jit_level为O0/O1模式下，支持亚健康设备热切换功能，无需重启训练集群即可替换亚健康节点，实现分钟级断点续训。用户可以通过设置环境变量 `export MS_ENABLE_TFT="{ARF:1}"` 开启该功能。

#### Parallel

- [STABLE] `msrun` 支持在启动分布式任务时，通过 `taskset` 命令对scheduler进程和worker进程进行绑核，用户可通过入参 `--bind_core` 对启动进程设置绑核。

#### PyNative

- [STABLE] 动态图下支持 `frombuffer`功能，允许从 Python 缓冲区协议对象（如 bytes 等）直接创建 MindSpore Tensor，无需数据拷贝。
- [STABLE] 动态图下支持 `saved_tensors_hooks` 功能，允许用户为在前向计算中被保留并在反向阶段使用的 saved tensor 配置自定义的保存和加载逻辑。

#### Python

- [STABLE] 支持Python 3.12版本。

### 非兼容性变更

CANN 8.5.0版本组合包完成开源开放架构升级，由Toolkit+ops两个组合包组成，解耦优化带来一系列命名/数量/使用关系及安装目录的变更，MindSpore 2.7.2版本需要与CANN 8.5.0配套使用，旧版本兼容将在MindSpore 2.8.0继续支持。

### 贡献者

Bellatan,caifubi,chaijinwei,changzherui,chengbin,fuchao,guangpengz,guozhijian,hhz886,huangbingjian,huda,jiangna,jiaorui,kingxian,liangchenghui,lichen,liubuyu,liuluobin,liuyanwei,looop5,maoyuanpeng1,Margaret_wangrui,mengxian,NaCN,panzhihui,r1chardf1d0,rogeryu11,shaoshengqi,shen_haochen,shenwei41,tanghuikang,wang_ziqi,wangshaocong,wangyin,xiaotianci,xuzhen,yanghaoran,yide12,yuchaojie,yuliangbin,zhangyinxia,zhaochenjie,zhengzuohe,ZPaC,zyli2020,曹彤,范吉斌,胡彬,宦晓玲,李良灿,熊攀

## MindSpore 2.7.1 Release Notes

### 主要特性及增强

#### Dataset

- [STABLE] 重构dataset `.map` 操作和 `.batch`操作多进程并行加速能力，将原有的 `c++ thread(with GIL) -> python process pool -> worker[i]`数据传递逻辑优化为 `c++ thread -> msg & shm -> worker[i]`数据传递，减少数据传递链路，解决GIL锁依赖，实现自定义数据处理效率显著提升。

#### Parallel

- [STABLE] 静态图模式场景，支持 `AlltoAllVC` 正反向算子，用户可通过 [mindspore.ops.AlltoAllVC](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.AlltoAllVC.html)接口使用该算子。动态图模式场景，支持通信接口[mindspore.communication.comm_func.all_to_all_v_c](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.comm_func.all_to_all_v_c.html)，用户可通过该接口使用该算子。
- [STABLE] 动态图模式场景，支持 `TCPStore`功能，支持通信类[mindspore.mint.distributed.TCPStore](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.TCPStore.html)，用户可通过该类使用 `TCPStore`功能。
- [STABLE] [mindspore.communication.create_group](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.create_group.html)接口支持传入参数 `hccl_comm`，从而复用外部创建通信组。用户可以用其他方式创建hccl通信组，然后传入给 MindSpore 组网。
- [BETA] [mindspore.mint.distributed.init_process_group](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.init_process_group.html) 接口支持传入参数 `init_method` 或 `store`，用户可通过该接口以不依赖scheduler进程的方式初始化通信。

#### Ascend

- [BETA] 动态图异步执行模式下支持算子自动融合功能，用户可通过设置环境变量 `export MS_DEV_PYNATIVE_FUSION_FLAGS="--opt_level=1"`开启。
- [BETA] 静态图场景支持CustomOpBuilder方式接入自定义算子，开放MS_CUSTOM_OPS_REGISTER自定义算子功能类注册宏。
- [BETA] 动态图场景支持 [mindspore.runtime.use_mem_pool](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.use_mem_pool.html) 方式传入自定义内存池，以实现在上下文中自定义内存分配。

#### PyNative

- [STABLE] [mindspore.mint.empty_like](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.empty_like.html)和[mindspore.mint.empty](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.empty.html)接口支持`pin_memory`参数，用户将`pin_memory`参数置为True，则返回的Tensor会分配到锁页内存，该特性只对CPU Tensor有效。
- [STABLE] View类算子性能优化。通过精简执行流程和优化关键实现，显著提升动态图（eager mode）下 View 算子的执行效率。
- [STABLE] [mindspore.Tensor.to](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.to.html)接口功能扩展，支持Tensor在Device之间拷贝数据。
- [STABLE] Tensor storage支持CPU/GPU/Ascend，在2.7.1之前Tensor Storage仅支持Ascend硬件。

#### Tools

- [STABLE] 静态图模式场景支持特征值联合检测方案，用户可通过配置环境变量MS_NPU_ASD_CONFIG启用功能。
- [STABLE] 支持HCCS链路故障检测，[stress_detect](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.utils.html#mindspore.utils.stress_detect)接口新增hccs检测类型。

### API 变更

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html) API新增了部分functional和Tensor接口。mint接口当前大多仍为实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式（图下沉）及CPU、GPU后端，后续会逐步完善。

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

- [STABLE] [mindspore.mint.nn.functional.conv1d](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.functional.conv1d.html)和[mindspore.mint.nn.Conv1d](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.Conv1d.html)接口从demo转为stable。
- [STABLE] [mindspore.mint.stack](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.stack.html)和[mindspore.mint.concat](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.concat.html)接口现在支持不同数据类型的Tensor输入。
- [BETA] [mindspore.ops.Morph](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Morph.html) 算子支持自定义反向传播函数（bprop），用户可为自定义正向计算定义相应的梯度计算逻辑。
- [BETA] [mindspore.recompute](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.recompute.html) 支持在静态图下使用，用户可以在@jit装饰的函数中调用该接口。
- [BETA] [mindspore.enable_dynamic](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.enable_dynamic.html) 支持符号推导功能，即其输入的 shape 允许使用[mindspore.Symbol](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Symbol.html)。

### 贡献者

Bellatan,caifubi,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,Dring,ehaleva,fary86,gaoyong10,guangpengz,GuoZhibin,guozhijian,haozhang,hedongdong,Henry Shi,hhz886,huangbingjian,huangzhuo,huangziling,huda,Huilan Li,huoxinyou,jiangshanfeng,jiaorui,jiaxueyu,jizewei,kairui_kou,kingxian,kisnwang,leida,liangchenghui,lichen,limingqi107,Linhai,LiNuohang,linux,liubuyu,liudongxu,liuluobin,liuyanwei,lizhitong,looop5,luochao60,machenggui,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,Metaqiang,NaCN,nepdada,panzhihui,Qiao_Fu,qiuleilei,qqqhhhbbb,qujianwei,shaoshengqi,shen_haochen,shenwei41,shuqian0,tanghuikang,uuhuu,wang_ziqi,wangjialin,wujueying,XianglongZeng,Xiaoda,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yanghaoran,yao_yf,yefeng,yide12,yiguangzheng,YijieChen,yuanqi,yuchaojie,YuJianfeng,YukioZzz,yuliangbin,yyyyrf,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhunaipan,ZPaC,zyli2020,范吉斌,龚昊宇,胡彬,宦晓玲,李栋,李良灿,李林杰,刘崇鸣,刘飞扬,刘力力,刘子涵,宋佳琪,孙昊辰,王泓皓,王振邦,俞涵,张栩浩,张学同,周一航

## MindSpore 2.7.0 Release Notes

### 主要特性及增强

#### Dataset

- [STABLE] [mindspore.dataset.vision.read_video](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.read_video.html) 接口新增支持基于Ascend VDEC的视频解码，用户可以通过 [mindspore.dataset.config.set_video_backend](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_video_backend.html) 接口指定视频处理后端为 `CPU`（默认）或 `Ascend` 。
- [STABLE] 新增 [mindspore.dataset.vision.VideoDecoder](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.VideoDecoder.html) 接口，提供视频抽帧解码及获取元数据的能力，当前仅支持 `Ascend` 后端。

#### 并行

- [STABLE] MindSpore 支持收集和规约分发非均匀大小的张量。用户可以通过 [mint.distributed.all_gather_into_tensor_uneven](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.all_gather_into_tensor_uneven.html) 和 [mint.distributed.reduce_scatter_tensor_uneven](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.reduce_scatter_tensor_uneven.html) 算子接口使用该功能。[mint.distributed.all_gather](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.all_gather.html) 和 [mint.distributed.reduce_scatter](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.reduce_scatter.html)也分别新增对非均匀大小的收集和规约分发操作支持。
- [BETA] 流水线并行支持ZeroBubbleV调度，降低流水线并行Bubble，同时可结合1B1F融合掩盖，提升通信和计算overlap的比例。
- [STABLE] MindSpore 优化 PP 通讯域，显著降低 HCCL_BUFFER_SIZE。
- [BETA] MindSpore 支持细粒度优化 `HCCL_BUFFER_SIZE`，可以通过环境变量 `MS_DEV_HCCL_CONF` 设置，详见[环境变量](https://www.mindspore.cn/docs/zh-CN/master/api_python/env_var_list.html)。

#### Compiler

- [BETA] 图模式支持 [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) 注册正反向 hook 的功能。

#### Runtime

- [STABLE] MindSpore支持预留大页内存。用户可以通过在 [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_memory.html) 接口中传入 `huge_page_reserve_size` 参数使能该功能。

### Lite

MindSpore Lite面向不同硬件设备提供轻量化AI推理加速能力，使能智能应用，为开发者提供端到端的解决方案，为算法工程师和数据科学家提供开发友好、运行高效、部署灵活的体验。

为了更好地促进人工智能软硬件应用生态繁荣发展，MindSpore Lite独立建仓促进生态发展。未来MindSpore Lite将与MindSpore AI社区一起，致力于丰富AI软硬件应用生态。

更多详情请参阅[MindSpore Lite代码仓](https://gitee.com/mindspore/mindspore-lite)。

### API 变更

- [STABLE] 作为[mindspore.mint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html) API接入任务的一部分，一些Tensor API的接口定义和功能进行了对齐和优化。

  | mindspore.Tensor                |
  | ------------------------------- |
  | mindspore.Tensor.masked_scatter |
  | mindspore.Tensor.bernoulli_     |
  | mindspore.Tensor.zero_          |
  | mindspore.Tensor.copy_          |

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html) API新增[mindspore.ops.ring_attention_update](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ring_attention_update.html)接口，当前仅支持 Atlas A2 训练系列产品。
- [STABLE] 新增[mindspore.enable_dynamic](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.enable_dynamic.html)接口，用于指定参数的shape是动态shape或者动态rank。

### 贡献者

Bellatan,caifubi,ccsszz,chaijinwei,chengbin,chenweifeng,chujinjin,DavidFFFan,DeshiChen,dingjinshan,fary86,fuchao,gaoyong10,GuoZhibin,guozhijian,haozhang,hedongdong,Henry Shi,hhz886,huangbingjian,huangziling,huda,Huilan Li,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,leida,lichen,limingqi107,LiNuohang,linux,liubuyu,liuluobin,looop5,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,NaCN,One_East,panzhihui,Qiao_Fu,qiuleilei,qiuyufeng,r1chardf1d0,SaiYao,shaoshengqi,shen_haochen,shenwei41,shuqian0,St.Universe,suteng,TAJh,tanghuikang,tianxiaodong,wang_ziqi,wangyibo,wujueying,wusimin,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yanghaoran,yangyingchun,yide12,yonibaehr,yuanqi,yuchaojie,YuJianfeng,YukioZzz,yuliangbin,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhaochenjie,Zhi Feng Wu,zhuguodong,ZPaC,zyli2020,程超,胡犇,胡彬,宦晓玲,黄勇,李良灿,李林杰,刘飞扬,刘勇琪,刘子涵,王振邦,熊攀,杨卉,俞涵,云骑士,张栩浩,周一航

## MindSpore 2.7.0-rc1 Release Notes

### 主要特性及增强

#### Ascend

- [STABLE] 新增 O1 场景下的 GroupedMatmul 优化，支持该算子后融合 element-wise 算子，大幅减少数据搬运开销，提升计算效率。用户可以通过设置环境变量 `MS_DEV_GRAPH_KERNEL_FLAGS`为 `"--enable_cluster_ops=GroupedMatmul"`开启。
- [STABLE] 提升内存 tracker 易用性：用户可以通过 [mindspore.runtime.memory_replay(file_path)](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.memory_replay.html) 导入 tracker 数据，用于模型显存分析；通过设置 `export MS_ALLOC_CONF=memory_tracker_path:file_path` 设置 tracker 数据保存路径；通过设置 `export MS_ALLOC_CONF=simple_tracker:true` 仅保存每个内存块最后一个使用者，减少保存数据大小。
- [STABLE] 优化了图模式下ops.Custom原语针对aclnn类型的自定义算子功能，全面支持非Tensor类型输入，并自动加载RegInfo信息，显著提升了aclnn自定义算子的易用性和灵活性。

#### PyNative

- [STABLE] [mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook) 和 [mindspore.nn.Cell.register_forward_pre_hook](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook) 新增 `with_kwargs` 参数（默认值：False），用于支持将 `construct` 调用时的关键字参数传递给 `hook_fn`。
- [STABLE] [mindspore.Tensor.register_hook](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook) 现已支持对多输出算子的输出 Tensor 注册 hook。
- [STABLE] Cell自定义反向功能增强，支持返回值的类型和对应输入的数据类型和shape形状的校验和自动转换。
- [STABLE] Tensor增加[Storage](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.storage.html?highlight=storage#mindspore.Tensor.storage)接口：支持通过Storage及其相关接口对内存进行操作，从而实现内存优化。
- [STABLE] 新增 C++ 的 [ms::pynative::PyboostRunner](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/cpp_api_for_custom_ops.html#class-pyboostrunner) 接口，方便自定义算子支持 PyNative 多级流水运行时。

#### 并行

- [STABLE] 针对通信域可配HCCL缓存：[communication.create_group](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) 支持传入通信域额外配置 `options`，通过 `hccl_config`键值对设置通信域HCCL通信域缓存大小，避免多通信域下OOM。
- [STABLE] 支持CPU通信接口 [mint.distributed.all_reduce](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.all_reduce.html)/[mint.distributed.barrier](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.barrier.html)/[mint.distributed.send](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.send.html)/[mint.distributed.recv](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.recv.html)，用户可通过这些接口使用对应的集合通信算子功能。
- [STABLE] 静态图模式场景，支持 [AllGatherV](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.AllGatherV.html) 和[ReduceScatterV](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ReduceScatterV.html)正反向算子。
- [BETA] 支持通过[mint.distributed.TCPStore](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.TCPStore.html)类的成员方法实现数据缓存功能。

#### 训练

- [STABLE] 重计算通信掩盖：支持对于完全重计算的两个Cell的计算通信进行互相掩盖，提升重计算场景模型的性能。
- [STABLE] 精度故障不重启快速恢复：支持当发生精度故障时，在不重启进程的情况下加载CheckPoint继续训练。
- [STABLE] 静默检测：支持对正反向计算中 `MatMul`结果进行校验。用户可以通过设置环境变量 `MS_SDC_DETECT_ENABLE` 为 `1` 使能，并通过 `mindspore.utils.sdc_detect` 模块的接口动态开启/关闭检测并获取结果。

#### 推理

- [STABLE] vLLM MindSpore 插件现已适配 vLLM v0.8.3 版本，并支持 vLLM V1 新架构的基础特性，包含 Chunked Prefill、Automatic Prefix Caching 等推理能力，在服务化部署方面，新增支持运行 DeepSeek-V3/R1  DP/TP/EP 混合并行推理能力，有效提升模型全量及增量推理效率，同时降低显存开销；

#### 工具

自2.3版本起MindInsight工具停止更新发布，相关文档已下架。原有系统调优数据可视化已整合至MindStudio Insight，标量可视化、参数分布图可视化和计算图可视化已整合至MindStudio Insight插件，请参阅[《MindStudio Insight用户指南》](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)。

- [STABLE] MindSpore Profiler支持接入msMonitor功能，支持用户通过在线监控工具采集性能数据。
- [STABLE] MindSpore Profiler增加record_shapes参数，支持用户采集框架侧下发算子的shape信息。
- [STABLE] MindSpore Profiler增加sys类资源类开关，支持sys资源类信息采集能力。
- [STABLE] MindSpore Profiler增加host_sys开关，支持系统调用类、存储类、cpu信息等host信息采集能力。
- [STABLE] MindSpore Profiler mstx模块提供domain功能，支持用户精细化控制mstx打点数据。

### API 变更

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html) API中部分DEMO状态的functional、nn和Tensor接口转为STABLE。mint接口当前大多仍为实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式(图下沉)及CPU、GPU后端，后续会逐步完善。

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

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html) API新增[mindspore.ops.swiglu](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.swiglu.html)接口，当前仅支持Ascend后端。
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html) API的[mindspore.ops.svd](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd)接口现在额外支持了Ascend后端。
- [STABLE] [mindspore.mint.nn.functional.silu](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.functional.silu.html#mindspore.mint.nn.functional.silu)和 [mindspore.mint.nn.SiLU](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.SiLU.html) 新增支持了入参 `inplace`。
- [STABLE] [communication.create_group](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) 新增支持通信域额外配置 `options`。HCCL后端支持设置 `options`中的 `hccl_config`，针对通信域设置HCCL通信域缓存大小。
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.runtime.html)增加[mindspore.runtime.empty_cache](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.empty_cache.html#mindspore.runtime.empty_cache)的实现。
- [STABLE] [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_memory.html#mindspore.runtime.set_memory)接口新增入参 `huge_page_reserve_size`。
- [STABLE] [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html#mindspore.runtime.set_cpu_affinity)接口新增入参 `module_to_cpu_dict`。
- [STABLE] [minspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)模块新增查看/保存模型状态的功能。新增接口：

  | mindspore.nn.Cell                       |
  | --------------------------------------- |
  | cell.register_state_dict_post_hook      |
  | cell.register_state_dict_pre_hook       |
  | cell.state_dict                         |
  | cell.register_load_state_dict_pre_hook  |
  | cell.register_load_state_dict_post_hook |
  | cell.load_state_dict                    |

- [STABLE] [minspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)模块查看/注册模型缓冲区的功能。新增接口：

  | mindspore.nn.Cell    |
  | -------------------- |
  | cell.register_buffer |
  | cell.get_buffer      |
  | cell.get_sub_cell    |
  | cell.named_buffer    |
  | cell.buffers         |

#### 非兼容性接口变更

- [runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)

  参数 `affinity_cpu_list`类型由字典变更为列表，仅针对单一进程自定义配置亲和CPU范围段。新增参数 `module_to_cpu_dict`，支持针对热点模块线程自定义配置CPU亲和策略。

  | 2.6     | 2.7       |
  | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, {"device0": ["10-19", "23-40"]}) | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, ["10-19", "23-40"],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {"main": [0,1,2,3],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "runtime": [4,5,6],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "pynative": [7,8,9]}) |

### 贡献者

baochong,Bellatan,BJ-WANG,caifubi,caiji_zhang,Carey,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,dingjinshan,Dring,ehaleva,Erpim,fary86,fengtingyan,fengyixing,fuchao,gaoyong10,gengdongjie,guangpengz,GuoZhibin,gupengcheng0401,haozhang,hedongdong,hhz886,huandong1,huangbingjian,huangziling,huda,HuilanLi,hujiahui8,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,jshawjc,kairui_kou,kingxian,kisnwang,lanzhineng,leida,LiangZhibo,lichen,limingqi107,LiNuohang,linux,liubuyu,liuchengji,liuluobin,liuyanwei,lkp,looop5,lujiale,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,NaCN,One_East,panshaowu,panzhihui,pengqi,Qiao_Fu,qiuleilei,qiuyufeng,rainyhorse,SaiYao,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shuqian0,tanghuikang,tangmengcheng,tan-wei-cheng,tianxiaodong,uuhuu,wang_ziqi,WangChengzhao,wangshaocong,wangyibo,wujueying,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yangguodong,yanghaoran,yangyingchun,Yanzhi_YI,yide12,yihangchen,YijieChen,yuanqi,yuchaojie,yuezenglin,YuJianfeng,YukioZzz,yuliangbin,yyuse,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhouyaqiang0,zhunaipan,zichun_ye,ZPaC,zyli2020,程超,范吉斌,胡犇,胡彬,宦晓玲,黄勇,李栋,李良灿,李林杰,李寅杰3,刘飞扬,刘力力,刘勇琪,刘子涵,梅飞要,宋佳琪,王泓皓,王禹程,王振邦,熊攀,徐安越,杨卉,杨明海,俞涵,虞良斌,云骑士,张栩浩,周一航

## MindSpore 2.6.0 Release Notes

### 主要特性及增强

#### Dataset

- [STABLE] [MindDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html)接口分片采样行为由原来的按块采样（链接中的数据分片的策略2）变更为间隔采样（链接中的数据分片的策略1），用户可以通过 MS_DEV_MINDRECORD_SHARD_BY_BLOCK 环境变量，控制是否切换回按块采样。
- [STABLE] GeneratorDataset 支持 spawn 方式启动多进程，支持在多进程时，使用Ascend后端的数据增强方法。用户可以设置 [mindspore.dataset.config.set_multiprocessing_start_method("spawn")](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_multiprocessing_start_method.html) 以 spawn 的方式启动多进程。
- [STABLE] [MindDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html) 的 `shuffle` 参数新增了 `Shuffle.ADAPTIVE` 行为，根据样本数量自适应调整 shuffle 样本数量的策略以降低训练内存开销，减少 OOM 风险。若期望强制采用全局 shuffle，可以指定为 `Shuffle.GLOBAL`，用户需确保机器内存足够。

#### Ascend

- [STABLE] 动态图模式场景，[ops.Custom](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Custom.html) 原语接入Ascend C自定义算子，支持多输出类型，`ops.Custom`支持C++侧infer type。
- [BETA] 动态图模式场景，新增CustomOpBuilder支持在线编译和加载自定义算子。
- [STABLE] 使用O1编译选项时，支持用户控制图算融合优化范围，用户通过环境变量MS_DEV_GRAPH_KERNEL_FLAGS的enable_fusion_pattern_only/disable_fusion_pattern选项，控制打开或者关闭对应融合pattern，同时支持通过--path=example.json方式读取文件配置。
- [STABLE] 支持通过 [mindspore.device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/zh-CN/master/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) 接口，设置aclop算子缓存信息老化配置和错误信息上报模式配置。
- [STABLE] GE后端仅支持整图下沉和lazy inline子图下沉，其他场景不再支持。
- [BETA] 静态图O0/O1模式场景，`mindspore.nn.Cell`基类新增offload接口与backward_prefetch接口属性，用户可通过[Cell.offload(backward_prefetch)](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.offload) 使用该接口，在训练正向阶段，将特定`Cell`类内的激活值从device侧卸载至host侧，并在训练反向阶段，将激活值从host侧提前预取回device侧。

#### Parallel

- [STABLE] 分布式pdb调试，支持动态图和静态图，更推荐使用动态图。
- [STABLE] 新增接口[mindspore.communication.get_comm_name](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.get_comm_name.html)，用户可以通过该接口查询HCCL集合通信库底层通信器名称。
- [STABLE] 新增 [AutoParallel](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html) 接口，支持对单个网络进行并行配置，解决并行配置作用域过大的问题。
- [STABLE] seqpipe新增支持两种调度方式seqvpp、seqsmartvpp，显著降低seqpipe结合vpp场景下的显存开销。
- [STABLE] 静态图模式场景，支持zero2/zero3级别的内存优化，降低有纯dp训练需求的模型的显存开销。
- [STABLE] 静态图模式场景，支持流水线并行下的1b1f通信掩盖，提升流水线并行性能。
- [STABLE] 静态图模式场景，支持张量模型并行和专家模型并行下的反向通信掩盖，提升模型训练性能。
- [STABLE] 静态图模式场景，自动并行策略传播模式更新为优先传播算子的Layout策略，提高策略传播准确性。
- [STABLE] 静态图模式场景，自动并行支持使用 [mindspore.parallel.shard](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.shard.html) 接口为mint算子配置策略，优化了多输入算子的策略。
- [STABLE]支持强化学习场景中，DP/MP/PP 多维混合并行模式下的训推权重在线权重重排。
- [STABLE] 支持用户查询分布式模块是否可用和通信模块是否初始化功能，用户可以通过 [mint.distributed.is_available](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.is_available.html) 接口查询分布式模块是否可用，以及通过 [mint.distributed.is_initialized](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.is_initialized.html) 接口查询通信模块是否初始化。
- [STABLE] 静态图模式场景，支持 `AlltoAllV` 正反向算子，用户可通过 [ops.AlltoAllV](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.AlltoAllV.html) 接口使用该算子。
- [STABLE] 支持CPU通信接口[mindspore.mint.distributed.allreduce](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.all_reduce.html#mindspore.mint.distributed.all_reduce)、[mindspore.mint.distributed.barrier](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.barrier.html#mindspore.mint.distributed.barrier)、[mindspore.mint.distributed.send](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.send.html#mindspore.mint.distributed.send)、[mindspore.mint.distributed.recv](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.distributed.recv.html#mindspore.mint.distributed.recv)，用户可通过这些接口使用对应的集合通信算子功能。

#### Inference

- [STABLE] 支持DeepSeek-V3/R1大模型的BFloat16全精度推理和W8A8量化推理，并为提升其推理性能开发或优化了RmsNormQuant、MatMul+Sigmoid+Add、Transpose+BatchMatMul+Transpose等12个融合算子。
- [BETA] 支持使用MindIE和MindSpore Transformers大模型套件，服务化部署DeepSeek-V3/R1。
- [STABLE] 优化了使用MindIE和MindSpore Transformers大模型套件进行推理服务部署时，加载safetensors的过程，实现了GE按需初始化，分别降低了内存占用量和启动耗时。
- [BETA] 支持使用[vLLM-MindSpore](https://gitee.com/mindspore/vllm-mindspore)插件和vLLM v0.6.6.post1，服务化部署DeepSeek-V3/R1、Qwen2.5大模型。

#### profiler

- [STABLE] 支持获取通信域并行策略信息，并行策略信息支持可视化显示，提升集群场景下性能定位效率。
- [STABLE] 动态profiling支持轻量化打点，用户可动态开启轻量化打点，实时查看性能数据。
- [STABLE] Profiler轻量化打点能力增强，支持dataloader、save checkpoint等关键阶段轻量化打点信息。
- [STABLE] Profiler支持查看memory_access相关aicore metric信息。
- [STABLE] Profiler支持[mindspore.profiler.profile](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.profile.html)、[_ExperimentalConfig](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler._ExperimentalConfig.html) 和[tensorboard_trace_handler](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html)，提升工具易用性。
- [STABLE] 动态profiling支持内存采集，用户可动态开启内存数据采集，提升工具易用性。

#### Compiler

- [BETA] 图模式支持inplace和view算子正向表达能力。
- [BETA] 图模式新增 [mindspore.ops.Morph](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Morph.html) 算子，允许将自定义函数封装成 [mindspore.ops.Morph](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Morph.html) 算子，易于分布式自动并行训练场景封装非规整集合通信（例如：[mindspore.ops.AlltoAllV](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.AlltoAllV.html)）。

### API 变更

#### 新增API

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html) API新增了大量的functional、nn接口。mint接口当前是实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式(图下沉)及CPU、GPU后端，后续会逐步完善。

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

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html) API 也提供了一些新增的stable接口。此外, 一些demo的接口也转为了stable。

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

- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API新增了大量的Tensor方法的接口。当前仍属于实验性接口，当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。详细见[官网接口列表](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)。
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html) API新增[mindspore.ops.moe_token_permute](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.moe_token_permute.html#mindspore.ops.moe_token_permute) 和 [mindspore.ops.moe_token_unpermute](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.moe_token_unpermute.html#mindspore.ops.moe_token_unpermute)两个推理算子接口，当前仅支持Ascend后端。
- [STABLE] [mindspore.mint.nn.functional.gelu](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.functional.gelu.html) 和 [mindspore.mint.nn.GeLU](https://www.mindspore.cn/docs/zh-CN/master/api_python/mint/mindspore.mint.nn.GELU.html) 新增支持了入参 "approximate"。
- [STABLE] 新增离线解析接口[mindspore.profiler.profiler.analyse](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.profiler.analyse.html)。

#### 非兼容性接口变更

- [mindspore.ops.Xlogy](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Xlogy.html) 接口入参input和other移除了对非Tensor输入的支持！[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  ops.Xlogy(input [Tensor, numbers.Number, bool],
            other [Tensor, numbers.Number, bool])
  </pre>
  </td>
  <td><pre>
  ops.Xlogy(input [Tensor],
            other [Tensor])
  </pre>
  </td>
  </tr>
  </table>

- `&`运算符在Ascend后端PyNative模式下不再支持uint32、uint64类型的Tensor输入，`^`运算符在Ascend后端PyNative模式下不再支持uint16、uint32、uint64类型的Tensor输入，`|`运算符在Ascend后端PyNative模式 `tensor | scalar`的场景下不再支持uint16、uint32、uint64类型的Tensor输入。[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)
- `%`运算符在CPU和GPU后端不再支持uint16、uint32、uint64类型的Tensor输入。[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)
- [mindspore.jit](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html) 接口参数变更。[(!80248)](https://gitee.com/mindspore/mindspore/pulls/80248)

  参数 `fn` 名称变更为 `function` 。

  移除参数 `mode` 、 `input_signature` 、 `hash_args` 、 `jit_config` 和 `compile_once` 。

  新增参数 `capture_mode` ，设置编译成MindSpore图的方式。

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

  新增参数 `jit_level` ，设置编译优化的级别。

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

  新增参数 `dynamic` ，设置是否需要进行动态shape编译。

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

  新增参数 `fullgraph` ，设置是否捕获整个函数来编译成图。

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

  新增参数 `backend` ，设置使用的编译后端。

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

  新增参数 `options` ，设置传给编译后端的选项字典。

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

- `mindspore.profiler.tensor_board_trace_handler` 接口变更。

  `mindspore.profiler.tensor_board_trace_handler`接口变更为 [mindspore.profiler.tensorboard_trace_handler](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html)。

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

- `mindspore.set_context`接口变更。

  参数 `ascend_config` 中的 `exception_dump`字段变更为 [device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/zh-CN/master/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) 中的 `"dump"`字段。

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

- `mindspore.Tensor`打印内容变更。

  原有Tensor打印内容，只打印值，新Tensor打印内容包含shape和dtype等Tensor关键信息。

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

- 静态图模式，Ascend后端，jit_level为O2模式下Dump接口变更。

  在静态图Ascend 后端 jit_level为O2场景下，环境变量 `MINDSPORE_DUMP_CONFIG`和 `ENABLE_MS_GE_DUMP`已废弃，Dump相关功能已迁移到 msprobe 工具，更多详情请查看[《msprobe 工具 MindSpore 场景精度数据采集指南》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md)。

### 贡献者

amyMaYun,Ava,baishanyang,br_fix_save_strategy_ckpt,caifubi,caoxubo,cccc1111,ccsszz,chaijinwei,chaiyouheng,changzherui,chengbin,chopupu,chujinjin,congcong,dairenjie,DavidFFFan,DeshiChen,dingjinshan,fary86,fengqiang,fengyixing,ffmh,fuhouyu,Gallium,gaoshuanglong,gaoyong10,geyuhong,guoyq16,guoyuzhe,GuoZhibin,guozhijian,gupengcheng0401,hangq,Hanshize,haozhang,hedongdong,hhz886,HighCloud,horcham,huangbingjian,huangxiang360729,huangzhichao2023,huangzhuo,huangziling,huda,Huilan Li,hujiahui8,huoxinyou,jiangchao_j,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,JoeyLin,jshawjc,kairui_kou,kakyo82,kisnwang,leida,lianghongrui,LiangZhibo,LiaoTao_Wave,lichen,limingqi107,LiNuohang,linux,litingyu,liubuyu,liuchuting,liuluobin,liuyanwei,LLLRT,looop5,luochao60,luojianing,luoxuewei,luoyang,lyk,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mylinchi,NaCN,panzhihui,pengqi,PingqiLi,pipecat,qiuyufeng,qiuzhongya,qiwenlun,r1chardf1d0,rachel0858,rainyhorse,Rudy_tan,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shiro-zzz,shuqian0,stavewu,TAJh,tanghuikang,tangmengcheng,tongdabao,TuDouNi,VectorSL,wang_ziqi,wangjie,wangliaohui97,wangpingan,wangyibo,weiyang,wja,wudongkun,wujueying,wuweikang,wwwbby,xfan233,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,xiedejin1,XinDu,xuxinglei,xuzhen,xuzixiang,yang guodong,yangben,yanghaoran,yangruoqi713,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yide12,yihangchen,YijieChen,yonibaehr,Youmi,yuanqi,yuchaojie,yuezenglin,Yuheng Wang,YuJianfeng,YukioZzz,ZeyuHan,zhaiyukun,Zhang QR,zhangbuxue,zhangdanyang,zhangshucheng,zhangyinxia,ZhangZGC,zhangzhen,zhengzuohe,zhouyaqiang0,zichun_ye,zlq2020,zong_shuai,ZPaC,zyli2020,舛扬,范吉斌,冯一航,胡犇,胡彬,宦晓玲,简云超,李栋,李良灿,李林杰,李寅杰3,刘思铭,刘勇琪,刘子涵,梅飞要,任新,十一雷,孙昊辰,王泓皓,王禹程,王振邦,熊攀,俞涵,虞良斌,云骑士,张栩浩,赵文璇,周一航

## MindSpore Lite 2.6.0 Release Notes

### 主要特性及增强

- [STABLE] MindSpore Lite支持模型转换时配置算子并行推理加速，只需在模型转换时配置stream_label_file选项，指定需要进行并行推理的算子。
- [STABLE] MindSpore Lite支持在昇腾后端下转换onnx控制流中的if算子。

### API 变更

- [STABLE] acl模型转换配置中，ascend_context选项下新增stream_label_file选项，用于启用多流并行。

### 贡献者

熊攀,ZhangZGC,yanghaoran,李林杰,shenwei41,xiaotianci,panzhihui,guozhijian,胡彬,tangmengcheng,XianglongZeng,cccc1111,stavewu,刘思铭,r1chardf1d0,jiangshanfeng

## MindSpore 2.5.0 Release Notes

### 主要特性及增强

#### 分布式启动组件msrun

- [STABLE] msrun支持传入节点的hostname（如localhost）作为 `--master_addr`，提升msrun易用性。
- [STABLE] msrun支持将训练日志打印到标准输出，用户可通过 `--tail_worker_log` 参数控制要打印哪些rank。
- [STABLE] 设置 `export VLOG_v=12500` 后， `scheduler` 日志能输出集群信息，帮助用户快速统计集群数据。
- [STABLE] msrun支持通过 `--worker_log_name` 参数来格式化日志文件名，帮助用户快速定位到问题节点。

详情可参考[msrun启动](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

#### Profiler

- [STABLE] 新增[mindspore.profiler.schedule](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.profiler.schedule.html)和[mindspore.profiler.tensor_board_trace_handler](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.profiler.tensor_board_trace_handler.html)接口，支持动态图场景按step采集和呈现，提升动态图场景易用性。
- [STABLE] 动态Profiling支持自定义for循环，提升动态图场景易用性。
- [STABLE] Profiler初始化参数和交付件目录结构优化，降低用户迁移难度。
- [STABLE] 新增轻量化打点接口[mindspore.profiler.mstx](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.profiler.mstx.html)，提供用户低开销性能数据采集方式。
- [STABLE] Timeline支持展示硬件利用率数据，帮助用户定位降频问题。

详情可参考[Ascend性能调优](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/optimize/profiler.html)。

#### 动态图

- [BETA] 动态图支持算子的原地操作，引入可以原地操作的算子。以[mindspore.mint.nn.functional.relu](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mint/mindspore.mint.nn.functional.relu.html)算子为列，如果你想使用原地更新版本的relu算子，则可以调用[mindspore.mint.nn.functional.relu_](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mint/mindspore.mint.nn.functional.relu_.html)算子。
- [STABLE] 使能环境变量MS_SIMULATION_LEVEL=1开启动态图dryrun，支持不占卡模拟多卡进程，通过日志查看显存占用情况。详情可参考[环境变量](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/env_var_list.html?highlight=ms_simulation_level#%E5%88%86%E5%B8%83%E5%BC%8F%E5%B9%B6%E8%A1%8C)。

#### FrontEnd

- [STABLE] 新增[mindspore.nn.utils.no_init_parameters](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/nn/mindspore.nn.utils.no_init_parameters.html)接口，支持网络参数延迟初始化，缩短推理场景下模型启动时间。

### API 变更

#### 新增API

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore.mint.html) API新增了大量的functional、nn接口。mint接口当前是实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式(图下沉)及CPU、GPU后端，后续会逐步完善。

  | mindspore.mint  |                           |                                   |                            |
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
  | mindspore.mint.meshgrid     | mindspore.mint.mm         | |                            |

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

- [STABLE] 新增[mindspore.ops.incre_flash_attention](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.incre_flash_attention.html)和[mindspore.ops.prompt_flash_attention](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.prompt_flash_attention.html)两个推理算子接口，当前仅支持Ascend后端。
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore.runtime.html)替代原mindspore.hal接口，提供了流、显存、Event等运行时资源相关的接口。
- [STABLE] [mindspore.device_context](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore.device_context.html)替代原set_context接口部分参数，提供了硬件平台相关的设置接口。
- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API新增了大量的Tensor方法的接口。当前仍属于实验性接口，当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。同时大量的存量Tensor接口，包括+=、-=、*=、/=等运算符，通过重载的方式在Ascend后端接入了Aclnn算子。详细见[官网接口列表](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)。

#### 非兼容性接口变更

- [mindspore.Tensor.new_ones](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/Tensor/mindspore.Tensor.new_zeros.html)接口的size入参取消对Tensor类型的支持。
- mindspore.Profiler删除参数timeline_limit、rank_id、analyse_only、env_enable。

- 接口名称：mindspore.Profiler

  变更内容：废弃profile_communication参数，通过设置profiler_level=ProfilerLevel.Level1或profiler_level=ProfilerLevel.Level2采集通讯矩阵数据。

  说明：profiler_level默认值为ProfilerLevel.Level0。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.5.0接口 </td>
  </tr>
  <tr>
  <td><pre>
  Profiler(profile_communication=True)
  </pre>
  </td>
  <td><pre>
  Profiler(profiler_level=ProfilerLevel.Level1)或Profiler(profiler_level=ProfilerLevel.Level2)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Profiler

  变更内容：废弃op_time参数，通过设置activaties=[mindspore.profiler.ProfilerActivity.NPU]设置采集NPU侧算子性能数据。

  说明：activaties参数类型为列表，只要包含mindspore.profiler.ProfilerActivity.NPU参数就表示使能采集NPU侧算子性能数据，默认开启采集。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.5.0接口 </td>
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

- 接口名称：mindspore.Profiler

  变更内容：aicore_metrics的参数类型由int改为mindspore.profiler.AicoreMetrics枚举值。

  说明：aicore_metrics默认值为mindspore.profiler.AicoreMetric.AiCoreNone。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.5.0接口 </td>
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

- 接口名称：mindspore.Profiler

  变更内容：废弃profile_framework参数，通过设置activaties=[mindspore.profiler.ProfilerActivity.CPU]采集框架测数据。

  说明：activaties参数类型为列表，只要包含mindspore.profiler.ProfilerActivity.CPU参数就表示使能采集框架性能数据，默认开启采集。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.5.0接口 </td>
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

### 贡献者

baishanyang ,bantao ,Bellatan ,biangelin ,BigSkySea ,caifubi ,candanzg ,candyhong ,Carey ,cccc1111 ,chaijinwei ,changzherui ,chengbin ,chengfeng27 ,chengxb7532 ,chujinjin ,coder2237 ,czrz ,dairenjie ,DavidFFFan ,DeshiChen ,dingjinshan ,ehaleva ,Erpim ,fary86 ,fengyixing ,ffmh ,fuchao ,fuhouyu ,gaoyong10 ,geyuhong ,guoyuzhe ,GuoZhibin ,guozhijian ,halo ,hangq ,haozhang ,hedongdong ,hehongzhe ,hhz886 ,HighCloud ,huangbingjian ,HuangLe02 ,huangziling ,huda ,Huilan Li ,hujiahui8 ,jiahaochen666 ,jiangchao_j ,jiangchenglin3 ,jiangshanfeng ,jiaorui ,jiaxueyu ,jizewei ,jjfeing ,JoeyLin ,jshawjc ,kakyo82 ,kingxian ,kisnwang ,leida ,liangchenghui ,lianghongrui ,LiangZhibo ,lichen ,limingqi107 ,LINH ,linux ,lionelchang ,lishanni ,liubuyu ,liujunzhu ,liuluobin ,liuxu ,liuyanwei ,liyan2022 ,LLLRT ,looop5 ,luochao60 ,luoxuewei ,luoyang ,lyk ,machenggui ,maoyuanpeng1 ,Margaret_wangrui ,master,mengxian ,MengXiangyu ,mengyuanli ,Mrtutu ,mylinchi ,NaCN ,Nikanuo ,niujunhao ,panzhihui ,pengqi ,PingqiLi ,pipecat ,qiuleilei ,qiuyufeng ,qiuzhongya ,r1chardf1d0 ,shaoshengqi ,shen_haochen ,shenhaojing ,shenwei41 ,shilishan ,shiro-zzz ,shuqian0 ,St.Universe ,stavewu ,superxf ,suteng ,TAJh ,tanghuikang ,tangmengcheng ,tan-wei-cheng ,tianxiaodong ,TuDouNi ,TYWZ22259 ,user_0145 ,VectorSL ,vincen45 ,wang_ziqi ,wangshaocong ,wangwensheng4 ,weiyang ,wtcheng ,wtobill ,wujiangming ,wujueying ,wuweikang ,wwwbby ,XianglongZeng ,xiaopeng ,xiaotianci ,xiaoyao ,xiedejin1 ,XinDu ,xuxinglei ,yang guodong ,yangben ,yanghaoran ,yanglong ,yanx ,Yanzhi_YI ,yao_yf ,yefeng ,Yi_zhang95 ,yide12 ,yihangchen ,YijieChen ,YingtongHu ,ylw ,yonibaehr ,yuanqi ,yuchaojie ,yuezenglin ,YuJianfeng ,yyuse ,Zhang QR ,zhangbuxue ,zhangdanyang ,zhanghaibo ,zhangminli ,zhangyinxia ,ZhangZGC ,zhangzhen ,zhengzuohe ,zhouyaqiang0 ,zhuguodong ,zichun_ye ,zlq2020 ,zong_shuai ,ZPaC ,zyli2020 ,陈一 ,程超 ,冯一航 ,胡彬 ,宦晓玲 ,黄勇 ,简云超 ,康伟 ,李栋 ,李良灿 ,李林杰 ,李寅杰,刘崇鸣 ,刘力力 ,刘思铭 ,刘涛Liu ,刘勇琪 ,刘子涵 ,吕浩宇 ,吕凯盟 ,梅飞要 ,倪轩 ,任新 ,十一雷 ,孙昊辰 ,王禹程 ,王振邦 ,熊攀 ,俞涵 ,虞良斌 ,张栩浩 ,赵文璇 ,周莉莉 ,周一航 ,邹文祥

## MindSpore 2.4.1 Release Notes

### 主要特性及增强

#### AutoParallel

- [STABLE] 支持split/concat分支通信计算并行，用户通过切分输入数据，形成可并行分支，分支间自动进行通信计算并行，降低通信开销。
- [STABLE] 支持Sequence pipeline，配套MindFormers的dev分支的LLama系列模型，通过引入Sequence维度拆分，降低流水线并行的Bubble以及显存开销。

#### PyNative

- [STABLE] PyNative模式通信算子默认按照通信域分配流，支持通信算子并发执行，协同并行策略优化，提供细粒度的通信掩盖，提升模型性能。

### 问题修复

- [IB0R4N](https://gitee.com/mindspore/mindspore/issues/IB0R4N)：修复在某些切分策略下，加载分布式权重精度不对的问题。

### 贡献者

bantao;caifubi;candanzg;chaijinwei;changzherui;chengbin;chujinjin;DeshiChen;dingjinshan;fary86;fuhouyu;gaoyong10;GuoZhibin;halo;haozhang;hedongdong;huangbingjian;hujiahui8;huoxinyou;jiangshanfeng;jiaorui;jiaxueyu;jshawjc;kisnwang;lichen;limingqi107;liubuyu;looop5;luochao60;luoyang;machenggui;MengXiangyu;Mrtutu;NaCN;panzhihui;qiuzhongya;shenhaojing;shilishan;tanghuikang;TuDouNi;wang_ziqi;weiyang;wujueying;XianglongZeng;xuxinglei;yang guodong;yanghaoran;yao_yf;yide12;yihangchen;YijieChen;YingtongHu;yuchaojie;YuJianfeng;zhangdanyang;ZhangZGC;zhengzuohe;zong_shuai;ZPaC;冯一航;胡彬;宦晓玲;李林杰;刘崇鸣;刘勇琪;任新;王禹程;王振邦;熊攀;俞涵;张栩浩;周一航;

## MindSpore 2.4.0 Release Notes

### 主要特性及增强

#### Dataset

- [STABLE] 修改 [mindspore.dataset.GeneratorDataset](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html)、[mindspore.dataset.Dataset.map](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.map.html)及 [mindspore.dataset.Dataset.batch](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html)接口中 `max_rowsize` 参数的默认值为None，以默认开启共享内存的动态分配，此时共享内存将随输入数据实时申请并加速数据处理，用户无需再事先调整该参数大小。
- [BETA] 数据处理支持独立进程模式，此模式下将减少训练进程与数据读取进程的GIL锁冲突，以提升动态图模式下的性能。可以通过环境变量 `MS_INDEPENDENT_DATASET`启动或关闭此模式。

#### Ascend

- [STABLE] 自定义算子支持昇腾动态图场景Pyboost执行模式，降低了算子调用开销。
- [STABLE] 昇腾Print算子支持输出超大tensor或print调用密集的场景，用户可以通过`MS_DUMP_SLICE_SIZE`和`MS_DUMP_WAIT_TIME`环境变量指定切片大小和超时时间以支持不同场景。
- [STABLE] 统一确定性计算设置，用户可以通过仅设置 `mindspore.set_context(deterministic="ON")`来使能昇腾确定性计算。
- [STABLE] 支持集合通信异常监控，监测到通信异常后，快速退出训练，避免超时等待。
- [STABLE] 支持[亚健康设备优雅退出](https://www.mindspore.cn/docs/zh-CN/r2.4.0/model_train/train_availability/graceful_exit.html)功能。训练框架检测到集群存在亚健康设备配置信息时，保存CKPT并统一结束集群训练进程。

#### Runtime

- [STABLE] O0/O1模式下支持后端编译缓存，前端编译缓存开启时默认开启。
- [STABLE] O0/O1模式下支持 aclnnAllGatherMatmul、aclnnMatmulReduceScatter 和 aclnnMatmulAllReduce 算子，提升性能。
- [STABLE] O0/O1模式下支持通过export MS_DISABLE_HEARTBEAT=1关闭集群心跳配置，降低Scheduler负载。
- [STABLE] O0/O1模式下支持通信算子融合。
- [STABLE] O2模式下支持虚拟内存，支持碎片整理功能，Ascend后端默认使能。
- [STABLE] 设备内存占用动态申请，支持单卡多用户使用，Ascend后端默认使能。
- [STABLE] O1模式下优化图算融合编译性能，默认使能。
- [STABLE] O1模式下支持kernel packet融合优化，提升动态shape网络执行性能，默认使能。
- [BETA] O1模式下支持MatMul后向融合（epilogue fuse）Elementwise算子。通过`mindspore.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")`使能。
- [BETA] O1模式下支持用户控制图算融合优化范围，用户通过graph_kernel_flags的enable_pass/disable_pass选项控制打开或者关闭对应融合算子。

#### PyNative

- [STABLE] [mindspore.nn.Cell.register_backward_hook](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook)/[mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook)对应的hook函数入参cell_id变更为cell的python对象。
- [STABLE] 新增[Cell.register_backward_pre_hook](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_pre_hook)接口，该API在Cell上注册反向传播的钩子函数，当每次计算完成该Cell的梯度时都会调用该钩子函数。
- [STABLE] 优化PyNative流程AICPU类算子下发缓存，提升API执行性能。
- [STABLE] 新增动态图下将一组Tensor占用的设备内存转换为一块连续的内存功能。

#### FrontEnd

- [STABLE] 在故障恢复场景，支持权重去冗余保存和加载。
- [STABLE] 混合精度训练，支持[auto模式](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/amp/mindspore.amp.auto_mixed_precision.html#mindspore.amp.auto_mixed_precision)。
- [STABLE] 支持对safetensors格式的保存、加载，以及并行场景下基于safetensors的离线汇聚和分布式加载。
- [BETA] 新增循环大算子接口 [mindspore.ops.WhileLoop](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.WhileLoop.html)、[mindspore.ops.ForiLoop](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.ForiLoop.html)、[mindspore.ops.Scan](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.Scan.html)，优化循环编译时间。
- [BETA] 图模式下支持算子传入关键字参数。

#### Parallel

- [STABLE] [mindspore.ops.TensorDump](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.TensorDump.html)算子支持分布式并行的场景，用户可通过配置TensorDump算子的 `input_output`属性决定打印输入/输出分片；新增接口[mindspore.ops.tensordump](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.tensordump.html)。
- [STABLE] msrun支持根据传入的rank table file来自定义rank id，支持通过 `--rank_table_file`传入的json文件来重排rank id。
- [STABLE] 支持昇腾单机内高性能通信库LCCL，用户可通过 `MS_ENABLE_LCCL` 环境变量在昇腾后端训练场景下使能LCCL通信库。
- [STABLE] 策略传播算法适配LLaMA/Mixtral类网络，减少用户配置LLaMA/Mixtral网络时切分策略的工作量。
- [STABLE] 支持高维张量并行,用户可通过配置[mindspore.ops.MatMul](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.MatMul.html)/[mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.BatchMatMul.html)算子的input_layout切换1D/2D/3D张量切分模式。
- [STABLE] 模拟编译在SIMULATION_LEVEL=0和SIMULATION_LEVEL=1运行方式jit_level为O0/O1时，不占用硬件资源。
- [STABLE] BatchMatMul模型并行引入的Allreduce在后续跟切分操作时，如果在parallel_speed_up_json中开启enable_allreduce_slice_to_reducescatter，根据匹配规则，自动化转换为ReduceScatter以减少通信量。
- [STABLE] [mindspore.nn.Cell.shard](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.shard)和[mindspore.shard](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.shard.html)支持用户配置mindspore.Layout类型的策略及各参数的切分策略parameter_plan。
- [BETA] SAPP支持在手工预配置算子并行切分策略后全自动生成剩余算子策略。用户通过打开 `MS_INTERFERED_SAPP` 环境变量来激活 `.shard()` 预配置的并行切分策略。
- [BETA] [mindspore.ops.Custom](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.Custom.html)算子支持配置切分策略。

#### Inference

- [STABLE] 新增Qwen2和LLaMA3.1系列大模型支持训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。
- [STABLE] 支持并行解码服务化部署，实现LLaMA系列大模型LookAhead投机推理。
- [BETA] 支持SLoRA服务化部署，实现大模型多微调权重调度推理。

#### Dump

- [STABLE] 优化[Dump文档](https://www.mindspore.cn/docs/zh-CN/r2.4.0/model_train/debug/dump.html)，按照设备类型和优化等级划分使用方式。
- [STABLE] Ascend O0/O1模式下支持异步Dump，包括异步Tensor、溢出、统计信息（host和device模式）。
- [STABLE] 溢出Dump支持配置最大溢出个数。
- [STABLE] Ascend O2模式下支持set dump。
- [STABLE] 支持qint4x2量化类型Dump。

### API 变更

#### 新增API

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore.mint.html) API新增了大量的functional、nn接口。mint接口当前是实验性接口，在图编译模式为O0和PyNative模式下性能比ops更优。当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。

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

#### 接口变更

- 接口名称：mindspore.dataset.GeneratorDataset

  变更内容：参数 `max_rowsize`默认值从 `6`变更为 `None`，以默认开启共享内存动态分配。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：参数 `max_rowsize`默认值从 `16`变更为 `None`，以默认开启共享内存动态分配。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.map

  变更内容：参数 `max_rowsize`默认值从 `16`变更为 `None`，以默认开启共享内存动态分配。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.ops.TensorDump

  变更内容：新增参数 `input_output`，控制打印行为。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：MindSpore Dump Tensor保存的文件格式

  变更内容：Dump得到的npy文件，会将原始Tensor的dtype信息添加到文件名中。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

#### 非兼容性接口变更

- 接口名称：mindspore.nn.Cell.register_backward_hook(hook_fn)

  变更内容：hook_fn的入参由cell_id变更为cell对象。

  说明：对原有hook，可以在hook_fn中通过id(cell)获取原有的cell_id。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
  </tr>
  <tr>
  <td><pre>
  def register_backward_hook(hook_fn)
  入参hook_fn(cell_id,
             grad_input, grad_output)
             -> New grad_output or None
  </pre>
  </td>
  <td><pre>
  def register_backward_hook(hook_fn)
  入参hook_fn(cell,
             grad_input, grad_output)
             -> New grad_input or None
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.nn.Cell.register_forward_hook(hook_fn)

  变更内容：hook_fn的入参由cell_id变更为cell对象。

  说明：对原有hook，可以在hook_fn中通过id(cell)获取原有的cell_id。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
  </tr>
  <tr>
  <td><pre>
  def register_forward_hook(hook_fn)
  入参hook_fn(cell_id, inputs, outputs)-> New outputs or None
  </pre>
  </td>
  <td><pre>
  def register_forward_hook(hook_fn)
  入参hook_fn(cell, inputs, outputs)-> New outputs or None
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.communication.comm_func.all_reduce

  变更内容：all_reduce新增入参async_op，返回值从Tensor变更为Tensor和CommHandle组成的tuple。

  说明：async_op表示all_reduce是否开启多流并行，默认值是False。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.all_gather_into_tensor

  变更内容：all_reduce新增入参async_op，返回值从Tensor变更为Tensor和CommHandle组成的tuple。

  说明：async_op表示all_gather_into_tensor是否开启多流并行，默认值是False。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.reduce_scatter_tensor

  变更内容：all_reduce新增入参async_op，返回值从Tensor变更为Tensor和CommHandle组成的tuple。

  说明：async_op表示reduce_scatter_tensor是否开启多流并行，默认值是False。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.isend

  变更内容：返回值从Tensor变更为Handle。

  说明：isend默认开启多流并行。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.irecv

  变更内容：返回值从Tensor变更为Handle。

  说明：irecv默认开启多流并行。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.all_to_all_with_output_shape

  变更内容：all_to_all_with_output_shape新增入参async_op，返回值从Tensor变更为Tensor和CommHandle组成的tuple。

  说明：async_op表示all_to_all_with_output_shape是否开启多流并行，默认值是False。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

- 接口名称：mindspore.communication.comm_func.all_to_all_single_with_output_shape

  变更内容：all_to_all_single_with_output_shape新增入参async_op，返回值从Tensor变更为Tensor和CommHandle组成的tuple。

  说明：async_op表示all_to_all_single_with_output_shape是否开启多流并行，默认值是False。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.4.0接口 </td>
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

### 贡献者

anyrenwei,bantao,baochong,Bellatan,BJ-WANG,caifubi,candanzg,candyhong,Carey,cccc1111,ccsszz,changzherui,chengbin,chengfeng27,chengxb7532,chenjianping,chenweifeng,chujinjin,dairenjie,DavidFFFan,DeshiChen,dingjinshan,emmmmtang,fanyi20,fary86,fengyixing,fix-dryrun,fuchao,fuhouyu,gaoyong10,gengdongjie,gent1e,GuoZhibin,guozhijian,halo,hangq,haozhang,hedongdong,Henry Shi,HighCloud,Hongxing,huandong1,huangbingjian,HuangLe02,huangziling,huda,huiliang166,hujiahui8,huoxinyou,jiangchenglin3,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jijiarong,jjfeing,JoeyLin,jshawjc,jxl,kairui_kou,kisnwang,kk,lanzhineng,LiangZhibo,lichen,limingqi107,lionelchang,liubuyu,liujunzhu,liuluobin,liyejun,LLLRT,looop5,luochao60,luoxuewei,luoyang,machenggui,maning202007,maoyuanpeng1,Margaret_wangrui,MengXiangyu,mengyuanli,moran,Mrtutu,mylinchi,NaCN,nomindcarry,panzhihui,paolopoggi,pengqi,pierreleca,qiuleilei,qiuyufeng,qiuzhongya,r1chardf1d0,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shihlCST,shilishan,shiro-zzz,shiziyang,shop-pin,shunyuanhan,shuqian0,stavewu,superxf,suteng,tanghuikang,tangmengcheng,tan-wei-cheng,tan-wei-cheng-3260,tianxiaodong,TronZhang,TuDouNi,VectorSL,vincen45,wang_ziqi,wanghenchang,wangjie,wangshaocong,weiyang,wtobill,wudawei,wujueying,wwwbby,xfan233,XianglongZeng,xiaotianci,xiaoxin_zhang,xiaoxiongzhu,xiaoxuanKL,xiaoyao,XinDu,xuxinglei,xuzhubin,yanghaoran,yanglong,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yefeng,yide12,yihangchen,YijieChen,YingLai Lin,ylw,yuanpeng2024,yuanqi,yuchaojie,Yuheng Wang,YuJianfeng,YukioZzz,yyuse,zangqx,ZeyuHan,zhangbuxue,zhanghaibo,zhangminli,zhangqinghua,zhangyanhui,ZhangZGC,zhangzhen,zhanzhan,zhengzuohe,zhouyaqiang0,zhuguodong,zichun_ye,zjun,zong_shuai,ZPaC,zuochuanyong,zyli2020,程超,蛋蛋de忧桑,狄新凯,范吉斌,冯一航,付国华,胡彬,宦晓玲,黄勇,黄卓,康伟,李良灿,李林杰,李寅杰3,刘崇鸣,刘思铭,刘涛Liu,刘勇琪,刘子涵,吕浩宇,吕昱峰（Nate.River）,钱丹,十一雷,孙昊辰,王禹程,王振邦,王梓润,吴大维,熊攀,徐安越,许子豪,俞涵,云骑士,张峻源,张王泽,张栩浩,赵文璇,周莉莉,朱家兴,邹文祥

## MindSpore 2.3.1 Release Notes

### 主要特性及增强

- [STABLE] 去掉了在使用 [Layout](https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/mindspore/mindspore.Layout.html#mindspore.Layout) 构建切分策略时，interleaved_parallel 对应的 device_matrix 的值必须为 2 的限制。
- [STABLE] 增加了用户自定义控制边的环境变量 [MS_CUSTOM_DEPEND_CONFIG_PATH](https://www.mindspore.cn/docs/zh-CN/r2.3.1/note/env_var_list.html)，用户可以自行配置控制边以实现更好的通信计算隐藏。

### API 变更

#### 新增API

- [STABLE] 新增API[mindspore.mint.repeat_interleave](https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/mint/mindspore.mint.repeat_interleave.html)。

### 贡献者

ccsszz;dairenjie;DeshiChen;fuhouyu;gaoshuanglong;gaoyong10;GuoZhibin;halo;huoxinyou;jiangchao_j;jiaorui;jiaxueyu;jijiarong;JuiceZ;lichen;liujunzhu;liuluobin;LLLRT;looop5;luoyang ;Margaret_wangrui;mengyuanli;panzhihui;pengqi;PingqiLi;Renyuan Zhang;tanghuikang;tianxiaodong;TuDouNi;wudawei;XianglongZeng;xiaosh;xiaoxin_zhang;XinDu;yanghaoran;yanglong;yangruoqi713;Yanzhi_YI;yao_yf;YijieChen;yuchaojie;YuJianfeng;zangqx;zhengzuohe;zhouyaqiang0;ZPaC;zyli2020;胡彬;宦晓玲;康伟;李林杰;刘崇鸣;王禹程;俞涵;周莉莉;邹文祥

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.3.1 Release Notes

### 主要特性及增强

昇腾后端模型转换时，支持使用配置文件中的[input_shape 参数](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/cloud_infer/converter_tool_ascend.html)来指定输入尺寸。

### API 变更

- [ModelGroup接口](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/cloud_infer/runtime_cpp.html) 新增模型权重共享支持，节省显存。
- [Model.get_model_info接口](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/converter_tool.html?highlight=get_model_info) 新增支持获取模型的输入尺寸。

### 贡献者

熊攀;ZhangZGC;jxl;zhangyanhui;emmmmtang;huandong1;yefeng

## MindSpore 2.3.0 Release Notes

### 主要特性及增强

#### AutoParallel

- [STABLE] 扩展函数式并行能力，[mindspore.shard](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.shard.html)新增支持图模式，图模式下以nn.Cell/function为单位设置输入与权重的并行切分策略，未设置的算子将通过"sharding_propagation"自动配置并行策略；增加支持手动重排布的[mindspore.reshard](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.reshard.html)接口，通过[mindspore.Layout](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.Layout.html)对张量设置精准切分策略。
- [STABLE] 新增Callback接口[mindspore.train.FlopsUtilizationCollector](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/train/mindspore.train.FlopsUtilizationCollector.html)统计模型算力利用率信息MFU和硬件算力利用率信息HFU 。
- [STABLE] 新增函数式通信接口[mindspore.communication.comm_func](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore.communication.comm_func.html)。
- [BETA] O0和O1模式下，优化interleaved pipeline的内存占用。
- [BETA] 自动并行模式下支持多机场景自动流水线策略生成（暂不支持单机场景自动流水线策略生成），需要将 `parallel_mode` 设置成自动并行 ``auto_parallel`` 并将 `search_mode` 设置成双递归算法 ``recursive_programming``。

#### PyNative

- [STABLE] 优化动态图的基础数据结构，提升算子API性能。
- [STABLE] Tensor支持[register_hook](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html)功能，以便用户打印或者修改Tensor对应的梯度。
- [STABLE] PyNative模式支持重计算功能，用户可以通过重计算接口降低网络的显存峰值。

#### FrontEnd

- [STABLE] 优化Checkpoint保存、加载基础流程，提升性能20%。
- [STABLE] 支持在保存、加载过程中对Checkpoint文件进行CRC校验，提升安全性。

#### Dataset

- [STABLE] 为以下数据增强增加昇腾处理后端支持：Equalize、Rotate、AutoContrast、Posterize、AdjustSharpness、Invert、Solarize、ConvertColor、Erase。
- [STABLE] 增加视频文件读取、解析功能支持，详见API：[mindspore.dataset.vision.DecodeVideo](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.DecodeVideo.html)、[mindspore.dataset.vision.read_video](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.read_video.html#mindspore.dataset.vision.read_video)、[mindspore.dataset.vision.read_video_timestamps](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/dataset_vision/mindspore.dataset.vision.read_video_timestamps.html#mindspore.dataset.vision.read_video_timestamps)。
- [STABLE] 支持在 `mindspore.dataset.GeneratorDataset`、`mindspore.dataset.Dataset.map` 及 `mindspore.dataset.Dataset.batch` 接口中指定 `max_rowsize` 参数为-1，此时数据多进程所使用的共享内存将随数据大小动态分配并高效运行，无需手动调参。

#### Inference

- [STABLE] 新增LLaMa2、LLaMa3、Qwen1.5等14个大模型支持训推一体架构，实现脚本、分布式策略和运行时的统一，典型大模型训练到推理部署周期下降到天级，通过融合大算子降低推理时延，有效提升网络吞吐量。

#### PIJIT

- [BETA] 支持Python 3.8和Python 3.10的字节码解析，扩大Python版本的支持范围。
- [BETA] 支持Dynamic Shape、Symbolic Shape作为输入，使能动态输入场景。
- [BETA] 使能单步构图能力，优化编译时间。
- [BETA] 通过调整字节码支持了带有副作用的字节码被捕获（STORE_ATTR、STORE_GLOBAL、LIST_APPEND、dict.pop），使能自动混合精度，减少裂图，提升性能。

#### Profiler

- [STABLE] 提供分级Profiler功能，通过profiler_level参数可控制按照不同级别进行性能数据采集。
- [STABLE] Profiler analyse方法新增mode参数，可配置异步解析模式，性能数据解析与训练并行。
- [STABLE] Profiler接口新增data_simplification参数，用户可控制性能数据解析完成后是否删除多余数据，节省硬盘空间。
- [STABLE] Profiler接口增强内存分析功能，用户通过profile_memory参数可采集框架、CANN、硬件的内存申请、释放信息，并可通过[MindStudio工具](https://www.hiascend.com/developer/blog/details/0230130822583032044)进行可视化分析。
- [BETA] PyNative模式下Timeline整合host profiling信息，包括任务耗时、用户侧堆栈信息。

#### Dump

- [STABLE] 增强同步和异步Dump功能，统计信息Dump新增L2Norm信息、新增statistic_category字段支持用户自定义需要保存的统计信息，提高Dump易用性。同步和异步Dump支持情况可参考[Dump功能说明](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/debug/dump.html#dump功能说明)。
- [STABLE] 完善同步Dump功能，通过配置op_debug_mode字段使能溢出和异常Dump。
- [STABLE] 增强同步Dump功能，通过配置stat_calc_mode字段可以使能device计算统计信息（默认在host计算），通过配置sample_mode字段可以进行采样Dump，提升Dump性能。
- [STABLE] 增强异步Dump功能，支持保存complex64和complex128格式。

#### Runtime

- [STABLE] 支持静态图多级编译，配置为[mindspore.set_context(jit_config={"jit_level": "O0/O1/O2"})](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.set_context.html)，默认值为空，框架根据产品类别自动选择优化级别，Atlas训练产品为O2，其余产品均为O0。
- [STABLE] 静态图O0/O1下支持通信计算多流并发执行。
- [STABLE] 新增[内存管理接口](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore.hal.html#内存管理)。
- [BETA] 内存池支持虚拟内存碎片整理，在静态图O0/O1下默认使能虚拟内存。

#### Ascend

- [STABLE] 提供昇腾平台上算子内存越界访问检测开关，用户可以通过设置 `mindspore.set_context(ascend_config={"op_debug_option": "oom"})`来检测昇腾平台上算子内部内存越界问题。
- [BETA] 环境变量[MS_SIMULATION_LEVEL](https://www.mindspore.cn/docs/zh-CN/r2.3.0/note/env_var_list.html)在昇腾平台上新增支持图编译O0执行模式，并可支持编译性能和运行时内存分析。
- [BETA] 昇腾平台支持通过AOT接入使用[AscendC自定义算子](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/operation/op_custom_ascendc.html)。

### API变更

#### 新增API

- [STABLE] 新增[mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore.mint.html)API，提供了大量的functional、nn、优化器接口，API用法及功能等与业界主流用法一致，方便用户参考使用。mint接口当前是实验性接口，在图编译模式为O0和PyNative模式下性能比ops更优。当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。

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

### 非兼容性接口变更

- 接口名称：性能数据采集接口 `Profiler`

  变更内容：解析生成的性能数据文件进行了精简，将在导出性能数据后删除FRAMEWORK目录数据以及其他多余数据，仅保留profiler的交付件以及PROF_XXX目录下的原始性能数据，以节省空间。通过将 `data_simplification`参数配置为 `False`可关闭精简模式，与历史版本生成的性能数据文件保持一致。
- 接口名称：Dump功能配置文件中的 `saved_data` 字段为 `"tensor"`。

  变更内容：Dump落盘的文件名发生变更，`"/"`用 `"_"`代替，算子名称变为算子全局名称。

  <table>
  <tr>
  <td style="text-align:center"> 原文件名 </td> <td style="text-align:center"> 2.3文件名 </td>
  </tr>
  <tr>
  <td><pre>
  文件名格式：
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.{format}.npy
  </br>
  示例：
  Conv2D.Conv2D-op12.0.0.1623124369613540.
  output.0.DefaultFormat.npy
  </pre>
  </td>
  <td><pre>
  文件名格式：
  {op_type}.{op_name}.{task_id}.{stream_id}.
  {timestamp}.{input_output_index}.{slot}.{format}.npy
  </br>
  示例：
  Conv2D.Default_network-WithLossCell__backbone-AlexNet_conv3
  -Conv2d_Conv2D-op12.0.0.1623124369613540.output.0.DefaultFormat.npy
  </pre>
  </td>
  </tr>
  </table>
- 接口名称：Dump功能配置文件中的 `saved_data`字段为 `"statistic"`。

  变更内容：原默认保存 `"max"`、`"min"`、`"avg"`、`"count"`、`"negative zero count"`、`"positive zero count"`、`"nan count"`、`"negative inf count"`、`"positive inf count"`、`"zero count"`、`md5`统计项，2.3变更为默认保存 `"max"`、`"min"`、`"l2norm"`统计项，可以通过配置 `statistic_category`自定义统计项。

### 贡献者

caifubi;candanzg;ccsszz;chaiyouheng;changzherui;chenfei_mindspore;chengbin;chengfeng27;Chong;dairenjie;DavidFFFan;DeshiChen;dingjinshan;douzhixing;emmmmtang;Erpim;fary86;fengyixing;fuhouyu;gaoyong10;GuoZhibin;guozhijian;halo;haozhang;hejianheng;Henry Shi;horcham;huandong1;huangbingjian;Jackson_Wong;jiangchenglin3;jiangshanfeng;jiangzhenguang;jiaorui;bantao;jiaxueyu;jijiarong;JuiceZ;jxl;kairui_kou;lanzhineng;LiangZhibo;lichen;limingqi107;linqingke;liubuyu;liujunzhu;liuluobin;liyan2022;liyejun;LLLRT;looop5;lujiale;luochao60;luoyang;lvxudong;machenggui;maning202007;Margaret_wangrui;master_2;mengyuanli;moran;Mrtutu;NaCN;nomindcarry;panzhihui;pengqi;qiuyufeng;qiuzhongya;Renyuan Zhang;shaoshengqi;Shawny;shen_haochen;shenhaojing;shenwei41;shij1anhan;shilishan;shiziyang;shunyuanhan;shuqian0;TAJh;tanghuikang;tan-wei-cheng;Thibaut;tianxiaodong;TronZhang;TuDouNi;VectorSL;wang_ziqi;wanghenchang;wangjie;weiyang;wudawei;wujiangming;wujueying;XianglongZeng;xiaotianci;xiaoxin_zhang;xiaoxiongzhu;xiaoyao;XinDu;xuxinglei;yangchen;yanghaoran;yanglong;yangruoqi713;yangzhenzhang;yangzishuo;Yanzhi_YI;yao_yf;yefeng;yide12;YijieChen;YingLai Lin;yuchaojie;YuJianfeng;zangqx;zhaiyukun;zhangminli;zhangqinghua;ZhangZGC;zhengxinQian;zhengzuohe;zhouyaqiang0;zhuguodong;zhupuxu;zichun_ye;zjun;zlq2020;ZPaC;zuochuanyong;zyli2020;阿琛;狄新凯;范吉斌;冯一航;胡彬;宦晓玲;黄勇;康伟;雷仪婧;李良灿;李林杰;刘崇鸣;刘力力;刘勇琪;刘子涵;吕浩宇;王禹程;熊攀;徐安越;徐永飞;俞涵;张王泽;张栩浩;郑裔;周莉莉;周先琪;朱家兴;邹文祥

欢迎以任何形式对项目提供贡献！

## MindSpore 2.3.0-rc2 Release Notes

### 主要特性和增强

#### AutoParallel

- [STABLE] Transpose/Sub/Add/Mul/Div/ReLU/Softmax/Sigmoid算子支持配置Layout。
- [STABLE] 集合通信精度会影响网络收敛，在接口mindspore.set_auto_parallel_context提供配置项[force_fp32_communication](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/mindspore/mindspore.set_auto_parallel_context.html)，设为True时可以强制将reduce类通信算子的通信类型转为float32。
- [BETA] pipeline并行支持Interleave调度，优化micro batch大小受限场景下的模型性能。
- [BETA] 优化pipeline并行场景下提高模型转换速度，支持单个stage单独转换。

#### PyNative

- [BETA] 动态图下支持[重计算](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/mindspore/mindspore.recompute.html)功能。
- [STABLE] 动态图下支持[register_hook](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook)功能。

### API变更

增加[动态组网](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc2/parallel/dynamic_cluster.html)场景下各类超时时间环境变量配置：

- `MS_TOPO_TIMEOUT`： 集群组网阶段超时时间，单位：秒。
- `MS_NODE_TIMEOUT`：节点心跳超时时间，单位：秒。
- `MS_RECEIVE_MSG_TIMEOUT`：节点接收消息超时时间，单位：秒。

新增环境变量 `MS_ENABLE_LCCL`，支持昇腾后端单机多卡场景下使用LCCL通信库。

### 问题修复

- [#I9CR96](https://gitee.com/mindspore/mindspore/issues/I9CR96) 修复在大规模集群下，动态组网启动方式的超时时间不足导致集群启动失败的问题。
- [#I94AQQ](https://gitee.com/mindspore/mindspore/issues/I94AQQ) 修复ops.Addcdiv算子在图模式下输出shape有误问题。

### 贡献者

感谢以下人员做出的贡献:

bantao,caifubi,changzherui,chenfei_mindspore,chenweifeng,dairenjie,dingjinshan,fangzehua,fanyi20,fary86,GuoZhibin,hanhuifeng,haozhang,hedongdong,Henry Shi,huandong1,huangbingjian,huoxinyou,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jxl,kairui_kou,lichen,limingqi107,liuluobin,LLLRT,looop5,luochao60,luojianing,maning202007,NaCN,niyuxin94520,nomindcarry,shiziyang,tanghuikang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wudawei,XianglongZeng,xiaoxiongzhu,xiaoyao,yanghaoran,Yanzhi_YI,yao_yf,yide12,YijieChen,YingLai Lin,yuchaojie,YuJianfeng,zangqx,zhanghanLeo,ZhangZGC,zhengzuohe,zhouyaqiang0,zichun_ye,zjun,ZPaC,zyli2020,冯一航,李林杰,刘力力,王禹程,俞涵,张栩浩,朱家兴,邹文祥

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.3.0-rc2 Release Notes

### 主要特性和增强

- [STABLE] 支持云侧转换工具所用的配置文件配置FlashAttention相关属性。
- [STABLE] 支持在多张卡上进行内存共享。

### 贡献者

感谢以下人员做出的贡献:

emmmmtang,熊攀

欢迎以任何形式对项目提供贡献！

## MindSpore 2.3.0-rc1 Release Notes

### 主要特性及增强

#### DataSet

- [STABLE] MindRecord模块增加完整性校验、加解密功能，以此保护用户数据的完整性与安全性。
- [STABLE] MindRecord接口变更：废弃FileWriter.open_and_set_header接口，因为其功能已内置到FilterWriter类，若使用旧版本代码将报错，删除此调用即可；FileWriter增加写入数据类型校验，以确保Schema定义的数据类型与真实数据类型匹配；Mindrecord组件下所有类方法去除返回值，若处理出错以异常方式提示用户。
- [STABLE] 为以下数据增强增加Ascend处理后端支持：ResizedCrop、HorizontalFlip、VerticalFlip、Perspective、Crop、Pad、GaussianBlur、Affine。
- [STABLE] 优化了模型迁移场景中数据迁移部分的指南，提供更多与第三方库框架对比的例子。
- [STABLE] 优化了TFRecordDataset在多数据列场景下解析效率，提升解析性能 20%。

#### PIJIT

- [BETA] PIJit通过对Python字节码进行分析&调整、执行流进行图捕获&图优化，支持的Python代码做静态图方式执行，不支持的进行子图切分以动态图方式执行，自动地做到动静统一。用户可以通过@jit(mode="PIJit", jit_config={options:value})对函数进行装饰来开启PIJit。

#### Inference

- [DEMO] 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，典型大模型训练到推理部署周期下降到天级，通过融合大算子降低推理时延，有效提升网络吞吐量。

#### AutoParallel

- [STABLE] 新增msrun启动方式，支持单指令拉起分布式任务。
- [STABLE] 添加RankTable启动方式即将废弃的提示。
- [STABLE] 图模式下消除冗余常量，提升编译性能和内存开销。
- [STABLE] 子图场景优化器并行首个子图inline，使得流水并行下的一些计算和通信掩盖可以进行。
- [STABLE] 通信信息导出，编译期间导出模型通信信息（通信域、通信量），输入给集群作为通信调度的依据。
- [STABLE] 流水线并行推理优化，去除共享权重在stage间转发，提升执行性能；支持流水线并行推理结果自动广播，提升自回归推理易用性。
- [STABLE] 算子级并行切分支持配置MatMul/Add/LayerNorm/GeLU/BiasAdd算子的切分时的设备排布与张量排布的映射关系。
- [STABLE] 支持数据并行维度的梯度通信与反向计算互相掩盖功能。
- [STABLE] 单卡模拟编译，用于模拟多卡分布式训练中某张卡的编译流程，辅助分析前后端各编译流程和内存占用。
- [STABLE] ops.Tril算子支持切分，从而降低对单个device的内存与性能需求。
- [BETA] 支持通信算子和计算算子融合，掩盖通信开销，提升网络性能。
- [BETA] 故障恢复时，checkpoint加载与编译并行从而减少故障恢复时间。

#### Runtime

- [BETA] 支持O0/O1/O2多级编译，提升静态图调试调优能力。

#### FrontEnd

- [STABLE] 框架新增对bfloat16数据类型的支持，创建Tensor时可以指定dtype=mindspore.bfloat16。
- [STABLE] 完善rewrite组件的语法支持能力，新增支持对类变量、函数、控制流等语法的解析。
- [STABLE] 新增context配置项：debug_level，用户可以使用mindspore.set_context(debug_level=mindspore.DEBUG)来获取更多调试信息。

#### Profiler

- [BETA] 动态启停profiling，用户可以根据训练情况实时采集profiling 数据，减少采集数据量。
- [BETA] Profiling通信算子耗时矩阵，用户通过分析通信算子耗时矩阵，找出集群通信性能瓶颈。
- [BETA] 提高昇腾环境解析Profiling数据的性能。
- [BETA] 支持离线解析Profiling生成的数据，用户可以先采集数据，然后根据需要再解析数据。
- [BETA] 支持采集片上内存、PCIe、l2_cache性能数据，丰富性能分析指标。

#### Dump

- [BETA] Dump保存的统计信息记录MD5值，用户可以通过MD5值确定张量值的微小差异。
- [BETA] Dump支持bfloat16数据类型，支撑用户定位bfloat16类型的算子精度问题。

#### PyNative

- [STABLE] 重构动态图下单算子调用流程，优化前端算子下发粒度，提升动态图性能。

#### Ascend

- [BETA] 支持用户设置CANN的options配置项，配置项分为global和session二类，用户可以通过mindspore.set_context(ascend_config={"ge_options": {"global": {"global_option": "option_value"}, "session": {"session_option": "option_value"}}})进行配置。

#### API变更

- 新增 mindspore.hal接口，开放流、事件以及设备管理能力。
- 新增 mindspore.multiprocessing 接口，提供了创建多进程的能力。

#### 算子

- [BETA] mindspore.ops.TopK当前支持第二个输入k为Int32类型的张量。

### 问题修复

- [#I92H93] 修复了昇腾平台下使用Print算子打印字符串对象时，Print算子报错Launch kernel failed的问题。
- [#I8S6LY] 修复了昇腾平台图模式动态shape流程下，变长输入算子（如 AddN、Concat）报错RuntimeError: Attribute dyn_input_sizes of Default/AddN-op1 is [const vector]{}, of which size is less than 0的问题。
- [#I9ADZS] 修复了故障恢复训练场景中，由于dataset恢复效率低导致网络训练出现数据超时的问题。

### 贡献者

感谢以下人员做出的贡献:

AlanCheng511，AlanCheng712，bantao，Bingliang，BJ-WANG，Bokai Li，Brian-K，caifubi，cao1zhg，CaoWenbin，ccsszz，chaiyouheng，changzherui，chenfei_mindspore，chengbin，chengfeng27，chengxb7532，chenjianping，chenkang，chenweifeng，Chong，chuht，chujinjin，Cynthia叶，dairenjie，DavidFFFan，DeshiChen，douzhixing，emmmmtang，Erpim，fangzhou0329，fary86，fengxun，fengyixing，fuhouyu，gaoshuanglong，gaoyong10，GaoZhenlong，gengdongjie，gent1e，Greatpan，GTT，guoqi，guoxiaokang1，GuoZhibin，guozhijian，hangq，hanhuifeng，haozhang，hedongdong，hejianheng，Henry Shi，heyingjiao，HighCloud，Hongxing，huandong1，huangbingjian，HuangLe02，huangxinjing，huangziling，hujiahui8，huoxinyou，jiangchenglin3，jianghui58，jiangshanfeng，jiaorui，jiaxueyu，JichenZhao，jijiarong，jjfeing，JoeyLin，JuiceZ，jxl，kairui_kou，kate，KevinYi，kisnwang，lanzhineng，liangchenghui，LiangZhibo，lianliguang，lichen，ligan，lihao，limingqi107，ling，linqingke，liruyu，liubuyu，liuchao，liuchengji，liujunzhu，liuluobin，liutongtong9，liuzhuoran2333，liyan2022，liyejun，LLLRT，looop5，luochao60，luojianing，luoyang，LV，machenggui，maning202007，Margaret_wangrui，MaZhiming，mengyuanli，MooYeh，moran，Mrtutu，NaCN，nomindcarry，panshaowu，panzhihui，PingqiLi，qinzheng，qiuzhongya，Rice，shaojunsong，Shawny，shenwei41，shenyaxin，shunyuanhan，silver，Songyuanwei，tangdezhi_123，tanghuikang，tan-wei-cheng，TingWang，TronZhang，TuDouNi，VectorSL，WANG Cong，wang_ziqi，wanghenchang，wangpingan，wangshaocong，wangtongyu6，weiyang，WinXPQAQ，wtcheng，wudawei，wujiangming，wujueying，wuweikang，wwwbby，XianglongZeng，xiaosh，xiaotianci，xiaoxin_zhang，xiaoxiongzhu，xiaoyao，XinDu，xingzhongfan，yanghaoran，yangluhang，yangruoqi713，yangzhenzhang，yangzishuo，yanjiaming，Yanzhi_YI，yao_yf，yefeng，yeyunpeng2020，yide12，YijieChen，YingLai Lin，YingtongHu，youshu，yuchaojie，YuJianfeng，zangqx，zby，zhaiyukun，zhangdanyang，zhanghaibo，zhanghanLeo，zhangminli，zhangqinghua，zhangyanhui，zhangyifan，zhangyinxia，zhangyongxian，ZhangZGC，zhanzhan，zhaoting，zhengyafei，zhengzuohe，ZhihaoLi，zhouyaqiang0，zhuguodong，zhumingming，zhupuxu，zichun_ye，zjun，zlq2020，ZPaC，zuochuanyong，zyli2020，陈宇，代宇鑫，狄新凯，范吉斌，冯一航，胡彬，宦晓玲，黄勇，康伟，李良灿，李林杰，刘崇鸣，刘力力，刘勇琪，吕浩宇，没有窗户的小巷，王禹程，吴蕴溥，熊攀，徐安越，徐永飞，许哲纶，俞涵，张峻源，张树仁，张王泽，张栩浩，郑裔，周莉莉，周先琪，朱家兴，邹文祥

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.13 Release Notes

### API变更

增加动态组网场景下各类超时时间环境变量配置：

- `MS_TOPO_TIMEOUT`： 集群组网阶段超时时间，单位：秒。
- `MS_CLUSTER_RETRY_NUM`：集群组网阶段节点重试注册次数。
- `MS_NODE_TIMEOUT`：节点心跳超时时间，单位：秒。
- `MS_RECEIVE_MSG_TIMEOUT`：节点接收消息超时时间，单位：秒。

### 问题修复

- [#I9CR96] 修复在大规模集群下，动态组网启动方式的超时时间不足导致集群启动失败的问题。

### 贡献者

感谢以下人员做出的贡献:

ZPaC, limingqi107, lizhenyu, jiangshanfeng

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.12 Release Notes

### 主要特性及增强

- [STABLE] 针对网络参数以fp32初始化以及开启优化器并行的场景，降低Cast算子数目。
- [STABLE] 增加对静默故障的检测和处理能力；静默故障会导致训练过程异常，该特性帮助用户避免或大幅降低因静默故障导致的集群停机巡检进行故障定位带来的损失。

### 问题修复

- [#I97D1L] 修复 ReduceLROnPlateau、LRScheduler、CosineAnnealingWarmRestarts动态学习率相关接口样例错误。
- [#I970HV] 修复多卡之间的allgather/reducescatter不保序问题。
- [#I99JPI] 修复checkpoint在模糊匹配场景下加载类型为bfloat16 parameter的 bug。

### 贡献者

感谢以下人员做出的贡献:

yao_yf, YijieChen, 冯一航, yuchaojie, 李良灿, YuJianfeng, huangxinjing, GuoZhibin, looop5

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.11 Release Notes

### 主要特性及增强

#### scipy

- [STABLE] 新增scipy模块API mindspore.scipy.optimize.linear_sum_assignment，用于解决线性和分配问题，它可以基于一个给定的成本矩阵，找到一个成本最低的分配方案。

### 问题修复

- [#I8JVRU] 修复bernoulli随机数算子在GPU上跑两次的结果出现概率性一致的问题。
- [#I8OC32] 修复MatrixSetDiagV3算子未校验异常输入，导致segmentation fault问题。

### 贡献者

感谢以下人员做出的贡献:

fary86, wanghenchang, haozhang, mengyuanli, emmmmtang, luoyang, zhupuxu, zhangyongxian, liuluobin, LLLRT, TuDouNi, hujiahui8, wangtongyu6, ligan, zhuguodong, yanghaoran, YingtongHu, liyejun, zjun, 徐永飞, chuht, 张树仁, 徐安越, DeshiChen, shenyaxin, liujunzhu, shunyuanhan, yuchaojie, yao_yf, 没有窗户的小巷, yeyunpeng2020, weiyang, KevinYi, hedongdong, zhouyaqiang0, Margaret_wangrui, zhanghaibo, moran, huangziling, 朱家兴, GuoZhibin, 李良灿, jiaxueyu, gaoyong10, Greatpan, 宦晓玲, melody, 俞涵, jiangshanfeng, XinDu, ling, caifubi, zhangyinxia, gengdongjie, Erpim, XianglongZeng, zhangminli, fengyixing, 冯一航, 黄勇, panzhihui, 胡彬, linqingke, wangshaocong

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.11 Release Notes

### 问题修复

- [#I8TPLY] 修复 SSD MobileNetV2 FPN 网络在Atlas 推理系列产品平台上的推理失败问题。

### 贡献者

感谢以下人员做出的贡献:

wangtongyu6, zhuguodong, 徐永飞, 徐安越, yeyunpeng2020, moran, XinDu, gengdongjie.

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.10 Release Notes

### 主要特性及增强

#### 算子

- [STABLE] FastGelu、BatchMatMul、AllReduce、AllGather、Broadcast、ReduceScatter算子支持bfloat16数据类型
- [STABLE] AllGather支持uint8数据类型

### 问题修复

- [#I8ALW3]修复Faster R-CNN、DeepTextMask、RCNN-ResNet50等网络在Ascend上8卡训练RandomChoiceWithMask算子报错问题
- [#I8LKG7]修复UNet-2D在Ascend 1卡、8卡图编译报错问题
- [#I8KU3X]修复CRNN-ResNet34在Ascend 1卡、8卡PyNative模式下训练进程卡住问题
- [#I8KTHH]修复在Ascend 8卡上使能enable_parallel_optimizer=True，不使用allreduce分组融合时，BERT网络训练报错问题

### 贡献者

感谢以下人员做出的贡献:

李林杰, TuDouNi, chengxb7532, Henry Shi, rms-infer-type, 朱家兴, zhouyaqiang0, tanghuikang, gaoyong10, gengdongjie, yao_yf, hujiahui8, hanhuifeng, shenyaxin, KevinYi, 冯一航, chengfeng27, JuiceZ, zhangyanhui, jijiarong, xiaoxiongzhu, 没有窗户的小巷, ling, liyan2022, haozhang, zangqx, xiaoyao, liujunzhu, 胡彬, panzhihui, wangshaocong, linqingke, jianghui58, qiuzhongya, yangruoqi713, zhangminli, moran, 王禹程, shaojunsong, wangtongyu6, zhupuxu, luoyang, 徐安越, qinzheng, caifubi, 徐永飞, chenkang, youshu, XinDu, liubuyu, jxl, yeyunpeng2020, huoxinyou, yefeng, jiaorui, wangpingan, cao1zhg, zjun, zyli2020, yanjiaming, Cynthia叶, 胡安东, 李良灿, liruyu, liuluobin, lihao, huangbingjian, YijieChen, jjfeing, looop5, 刘力力, xiaoxin_zhang, yangluhang, chenweifeng, jiangshanfeng, zichun_ye, 陈宇, NaCN, ligan, YingLai Lin, huangziling, chenjianping, DeshiChen, chengbin, kairui_kou, ccsszz, yanghaoran, zhangdanyang, Yanzhi_YI, zhengzuohe, hangq, TronZhang, wanghenchang, HighCloud, 吕浩宇, VectorSL, ZPaC, mengyuanli, maning202007, 刘勇琪, r1chardf1d0, fary86, 刘崇鸣, yuchaojie, douzhixing, fengyixing

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.10 Release Notes

### 问题修复

- [#I8K7CC]优化get_model_info接口传入非str字段的报错

### 贡献者

感谢以下人员做出的贡献:

gengdongjie, zhangyanhui, xiaoxiongzhu, wangshaocong, jianghui58, moran, wangtongyu6, 徐安越, qinzheng, 徐永飞, youshu, XinDu, yeyunpeng2020, yefeng, wangpingan, zjun, 胡安东, 刘力力, 陈宇, chenjianping, kairui_kou, zhangdanyang, hangq, mengyuanli, 刘崇鸣

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.1 Release Notes

### Bug Fixes

- [#I7R3R5] 修复昇腾平台ResNet-50网络精度劣化问题。
- [#I8A9RH] 修复昇腾平台DBNet（ResNet-50）网络精度劣化问题。
- [#I8B8IW] 修复多维Tensor赋值越界导致段错误的问题。
- [#I8J0F4] 修复多维Tensor扩展维度在动态图执行失败的问题。
- [#I87P3P] 修复昇腾平台二次训练编译缓存加载失败的问题。
- [#I86GP9] 修复昇腾平台UNet3D网络推理精度劣化问题。
- [#I89B4K] 修复Windows平台动态图动态rank执行卡住的问题。
- [#I8CX0C] 修复昇腾平台上动态图混合精度模式下偶现失败的问题。
- [#I8BGCF] 修复昇腾平台AIRNet网络动态图模式下执行出现段错误的问题。
- [#I8L5DS] 修复昇腾平台ResNet-50图像分割网络动态图执行慢的问题。

### 贡献者

感谢以下人员做出的贡献:

yufan, dingcheng, lvzhangcheng, zhunaipan, fangwenyi, weiyang, changzherui, chujinjin, zangqingxiang, yuchaojie, wuweikang, tanghuikang, xiaoyao, huangbinjian, zhoupeichen, chenfei_mindspore, hedongdong, wangnan, zhengzuohe, yanghaoran, zouliqin, luoyang, liuchongmin, lujiale, machenggui, wangcong, lixiangyi, wangting, huangyong

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.1 Release Notes

### Bug Fixes

- [#I88055] 修复MindSpore Lite推理gridsample算子format设置错误的问题。
- [#I8D80Y] 修复MindSpore Lite推理单算子调用流程资源释放异常的问题。

### 贡献者

感谢以下人员做出的贡献:

zhanghaibo, wangsiyuan, yefeng, wangshaocong, chenjianping

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.0 Release Notes

### 主要特性和增强

#### DataSet

- [STABLE] 数据操作map/batch的`row_size`参数扩展支持传入list，代表[输入共享内存, 输出共享内存]，以便在多进程模式时灵活控制共享内存的大小。
- [STABLE] 为官网API文档页面mindspore.dataset、mindspore.dataset.transforms、mindspore.mindrecord的所有API补充完善样例，方便用户参考。
- [STABLE] ConcatDataset支持全局采样能力，即使用concat操作组合多来源数据后，可以对数据进行全局随机采样以增强数据多样性。
- [STABLE] 使用model.train接口训练时，支持通过TimeMonitor(.., data_time=True)实时监控数据处理性能。
- [STABLE] 引入jemalloc库，解决在极端场景下，因内存碎片回收不及时导致内存缓慢上涨问题。

#### FrontEnd

- [STABLE] 支持添加@lazy_inline装饰器来标注Cell生成的子图延迟inline，从而有效提升编译性能。
- [STABLE] 新增CellDict数据结构，支持构建Dict类型的Cell对象，完善构建网络能力。
- [STABLE] 混合精度训练的功能优化，支持通过rewrite自动改写python脚本实现混合精度策略，支持函数、分支语句等多种语法自动解析。
- [STABLE] 动态学习率功能优化，新增MultiStepLR等API；get_lr方法与global_step解耦，扩展优化器模块功能。
- [STABLE] 优化API代码样例、API差异表以及高阶函数使用教程。

#### 算子

- [STABLE] 新增算子原语`mindspore.ops.Dense`。
- [STABLE] 新增随机数算子状态管理功能，使随机数算子可以保存随机数状态，并在模型并行、重计算等场景稳定复现。当前仅支持CPU/GPU平台，涉及的随机数算子包括：`mindspore.ops.Multinomial`、`mindspore.ops.MultinomialWithReplacement`、`mindspore.ops.ParameterizedTruncatedNormal`、`mindspore.ops.StandardLaplace`、`mindspore.ops.StandardLaplace`、`mindspore.ops.Uniform`、`mindspore.ops.UniformInt`、`mindspore.ops.UniformReal`、`mindspore.ops.UniformInt`、`mindspore.ops.Dropout`、`mindspore.ops.RandomChoiceWithMask`、`mindspore.ops.RandomCategorical`、`mindspore.ops.RandomShuffle`、`mindspore.ops.RandamGamma`、`mindspore.ops.RandomPoisson`、`mindspore.ops.TruncatedNormal`。
- [STABLE] 当GPU算子遇到非法输入场景，支持在算子的CUDA核函数中异步打印报错日志到Host侧，并中断当前CUDA Stream的执行，提高用户算子问题的定位效率。

#### PyNative

- [STABLE] PyNative模式下支持View机制。
- [STABLE] PyNative模式下功能增强：sens支持dict类型输入。

#### Ascend

- [STABLE] 支持用户可配置算子高精度/高性能模式，用户可以通过`mindspore.set_context(ascend_config={"op_precision_mode": "/path/to/op_precision_config_file"})`对部分TBE算子配置高精度/高性能模式。
- [BETA] 支持用户可配置fp16进fp32出的算子，用户可以通过`mindspore.set_context(ascend_config={"precision_mode": "force_fp32"})`对TBE Cube算子配置fp16进fp32出。
- [BETA] 去除jit level "O3"与GE流程强绑定，用户在执行GE流程时无需再设置`jit_level="O3"`。

#### Parallel

- [STABLE] 支持半自动/全自动模式下，非流水线并行场景的梯度累加特性，用户可以通过`net = GradAccumulationCell(net, micro_size)`方式，对网络使能梯度累加。梯度累加特性同样支持LazyInline编译加速。

#### 推理

自2.2版本起MindSpore主发布包不再提供配套310的推理接口使能，如需使用请切换安装MindSpore Lite发布包或下载MindSpore2.0之前的版本。MindSpore lite的安装部署与用法详见 <https://www.mindspore.cn/lite>。昇腾（Ascend）310是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式模型进行推理。原先MindSpore提供了两种在Ascend 310硬件上的推理使能用法：

1. 由MindSpore主发布包提供配套Ascend 310的版本，支持C++推理接口。
2. 由MindSpore Lite发布包提供配套Ascend的版本，支持C++/Java两种语言进行推理。

这两种方案提供的C++ API基本一致，后续不再构建和维护两套接口，而是归一使用MindSpore Lite。原有基于MindSpore主发布包构建的310推理业务，可以少量修改切换到MindSpore Lite，详见 <https://www.mindspore.cn/docs/zh-CN/master/faq/inference.html>。

### Bug fixes

- [I7SDA0] 修复了昇腾平台上CRNN网络精度劣化的问题。
- [I7T4QK] 修复了昇腾平台上wgan网络推理精度劣化问题。
- [I7TJ8Z] 修复了昇腾平台上lgtm网络推理精度劣化问题。
- [I7M58O] 修复了昇腾平台上ASR-dynamic网络训练core-dump的问题
- [I7L6B6] 修复了dataset多进程模式时，子进程在某些场景不退出的问题。
- [I7L7AE] 修复了dataset处理中包含repeat操作，且dataset.batch中使用动态batch时，batchinfo.get_epoch_num()计算不正确的问题。
- [I7UY7G] 修复OBSMindDataset中对于文件权限修改的异常的报错。

### 贡献者

感谢以下人员做出的贡献:
bantao, Bingliang, BJ-WANG, Brian-K, caifubi, ccsszz, changzherui, chenfei_mindspore, chengfeng27, chenhaozhe, chenjianping, chenkang, chenweifeng, chuht, chujinjin, CShu0507, Cynthia叶, DeshiChen, douzhixing, Erpim, Etienne, fary86, fengxun, fengyixing, gaoshuanglong, Gaoxiong, gaoyong10, GaoZhenlong, Greatpan, GuoZhibin, guozhijian, hangq, hanhuifeng, haozhang, hedongdong, Henry Shi, HighCloud, Hongxing, huangbingjian, huanghui, huangxinjing, huangziling, hujiahui8, huoxinyou, HWalkingMan, jianghui58, jiangshanfeng, jiaorui, jijiarong, jjfeing, JuiceZ, jxl, KevinYi, kisnwang, KXiong, lanzhineng, Li Qingguo, LiangZhibo, lianliguang, ligan, lihao, Lihoon, limingqi107, ling, linqingke, liruyu, liubuyu, liuchao, liujunzhu, liuluobin, liupeng303, liutongtong9, liyan2022, liyejun, looop5, luochao60, luojianing, luoyang, machenggui, maning202007, Margaret_wangrui, MaZhiming, mengyuanli, moran, NaCN, nomindcarry, panshaowu, panzhihui, qinzheng, qiuzhongya, r1chardf1d0, shaojunsong, shenwei41, shenyaxin, shenzhangyi, Shira Zaloshinski, shunyuanhan, tangdezhi_123, tanghuikang, tan-wei-cheng, tan-wei-cheng-3260, TronZhang, TuDouNi, VectorSL, wang_ziqi, wanghenchang, wangpingan, wangshaocong, wangtongyu6, wtcheng, wujueying, XianglongZeng, xiaotianci, xiaoxin_zhang, xiaoxiongzhu, xiaoyao, xiaoyuanyuan, XinDu, xujinliang, xupan, yanghaoran, yangluhang, yangruoqi713, yangsijia, yangzhenzhang, yangzishuo, yanjiaming, Yanzhi_YI, yao_yf, yefeng, yeyunpeng2020, yide12, YijieChen, YingLai Lin, YingtongHu, yonibaehr, youshu, yuchaojie, YuJianfeng, zangqx, zhaizhiqiang, zhangbuxue, zhangchunlei, zhangdanyang, zhangdong, zhanghaibo, zhangminli, zhangqi, zhangqinghua, zhangyanhui, zhangyifan, zhangyongxian, zhangzhen, zhangzheng, zhanzhan, zhengzuohe, ZhihaoLi, zhoufeng, zhouyaqiang0, zhuguodong, zhupuxu, zichun_ye, zjun, ZPaC, zuochuanyong, zyli2020, 陈宇, 程超, 范吉斌, 冯浩, 冯一航, 胡彬, 宦晓玲, 黄勇, 雷元哲, 黎冠新, 李良灿, 李林杰, 刘崇鸣, 刘力力, 刘思铭, 刘勇琪, 吕浩宇, 没有窗户的小巷, 沈竞兴, 王禹程, 王振邦, 徐安越, 徐永飞, 俞涵, 张澍坤, 周超, 朱家兴

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.0 Release Notes

### 主要特性和增强

#### 支持FlashAttention算子融合

- [STABLE] 在Ascend系列硬件上，支持LLAMA、stable diffusion系列模型的FlashAttention大算子融合。

## MindSpore 2.1.1 Release Notes

### Bug fixes

- [I7Q9RX] 昇腾平台支持不同硬件类型自适应识别。
- [I7SDA0] 修复了昇腾平台上CRNN网络精度劣化的问题。
- [I7T4QK] 修复了昇腾平台上wgan网络推理精度劣化问题。
- [I7TJ8Z] 修复了昇腾平台上lgtm网络推理精度劣化问题。

### 贡献者

感谢以下人员做出的贡献:

changzherui, chenfei_mindspore, chenjianping, chenkang, chenweifeng, chujinjin, fangwenyi, GuoZhibin, guozhijian, hangq, hanhuifeng, haozhang, hedongdong, 尤澍, zhoufeng, 代宇鑫

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.1.1 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite Cloud Inference adds support for Python 3.8 and Python 3.9

## MindSpore 2.1.0 Release Notes

### 主要特性和增强

#### FrontEnd

- [BETA] JIT Fallback支持变量场景：静态图模式下，支持返回Dict类型和Scalar类型，支持对非Parameter类型对象进行属性设置， 支持List的部分就地修改操作，完善支持NumPy等第三方库，支持用户自定义类的相关操作，支持Python基础运算符、内置函数使用更多数据类型，兼容控制流、副作用、自动微分等功能。具体用法请参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r2.1/note/static_graph_syntax_support.html)。
- [BETA] 静态图模式下，优化控制流场景中使用未定义变量的报错。使用if、while、for控制流分支内定义的变量，需在控制流之前初始化定义变量。
- [STABLE] 新增ReWrite功能，支持基于自定义规则修改网络结构，提供对多个网络进行批量修改的能力。
- [BETA] 新增optim_ex优化器模块，扩展现有功能，支持全量优化器参数分组策略的设置、支持运行中通过赋值的方式修改参数等功能。
- [STABLE] 优化MindSpore与PyTorch的API映射表，详细介绍API在功能、参数、输入、输出和特定场景等方面的差异。

#### PyNative

- 优化动态图模式下动态shape场景的性能。

#### DataSet

- [STABLE] 优化MindRecord数据文件的内存结构，加载百TB级别数据训练可降低60%内存占用。
- [STABLE] 支持单线程执行数据处理Pipeline，以便用户在数据Pipeline中添加代码对数据处理功能进行调试。
- [STABLE] 优化了TFRecordDataset的性能，对数据集加载性能提升60%+；优化了batch的性能，对于batch数较大的使用场景性能提升30%。
- [STABLE] 优化API文档[mindspore.dataset](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.html) 和 [mindspore.dataset.transforms](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.transforms.html)的Example示例，并新增了四篇样例库展示数据增强的效果，分别是：[使用数据Pipeline加载 & 处理数据集](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.html#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86pipeline%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B)、[视觉变换样例库](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.transforms.html#%E6%A0%B7%E4%BE%8B%E5%BA%93)、[文本变换样例库](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.transforms.html#%E6%A0%B7%E4%BE%8B%E5%BA%93-1)、[音频变换样例库](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.transforms.html#%E6%A0%B7%E4%BE%8B%E5%BA%93-2)

#### AutoParallel

- [STABLE] 支持训练过程将参数或者中间结果offload到CPU或NVMe，用户通过配置context开启自动offload功能，扩大可训练模型规模。

- [STABLE] 自动并行能力增强：

  1. 典型网络自动策略性能不低于默认配置的90%；

  2. 支持3D混合并行训练：自动算子级策略生成结合手动配置pipeline切分。

#### Runtime

- [STABLE] 升级OpenMPI版本至4.1.4。
- [STABLE] 升级NCCL版本至2.16.5。
- [STABLE] 动态组网场景下单节点内多卡rank连续分配。
- [STABLE] 动态组网场景下用户无需在脚本中对Scheduler角色进行适配，Scheduler与Worker脚本可保持完全一致。

#### Ascend

- [STABLE] 算子执行发生AIC Error时日志支持输出辅助AIC Error定位的维测信息，信息包括算子task名字、stream id、输入输出及workspace地址等。
- [STABLE] 针对算子输出为空Tensor的场景为CANN算子提供默认的处理机制，即跳过其算子执行。
- [STABLE] 在图模式网络模型执行失败时补充相关定位信息，即在rank_${id}/exec_order/目录下产生csv文件，记录每个task的task id和stream id。

#### Profiler

- [STABLE] Profiler支持收集Host侧各个阶段耗时数据。
- [BETA] Profiler支持收集Host侧各个阶段内存数据。
- [BETA] Profiler支持收集数据处理算子耗时。

### API变更

- `mindspore.set_context`新增`jit_syntax_level`选项，用于设置JIT语法支持级别，请参考[set_context](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore/mindspore.set_context.html)。
- 变更了`model.infer_predict_layout`接口，接口新增参数skip_backend_compile，默认值为False。当用户希望跳过后端编译流程获取参数切分策略时可选择设置为True。

#### 算子

- 新增算子原语`mindspore.ops.ApplyAdamWithAmsgradV2`，推荐通过接口`mindspore.nn.Adam`调用。
- 新增算子原语`mindspore.ops.UpsampleTrilinear3D`，推荐通过接口`mindspore.ops.interpolate`调用。
- 新增算子原语`mindspore.ops.UpsampleNearest3D`，推荐通过接口`mindspore.ops.interpolate`调用。

#### 接口弃用

- 弃用算子原语`mindspore.ops.ScatterNonAliasingAdd`，推荐使用算子原语`mindspore.ops.TensorScatterAdd`替换。

#### 非兼容性接口变更

- 接口名称：`mindspore.nn.Dense`、`mindspore.nn.Conv1d`、`mindspore.nn.Conv1dTranspose`、`mindspore.nn.Conv2d`、`mindspore.nn.Conv2dTranspose`、`mindspore.nn.Conv3d`、`mindspore.nn.Conv3dTranspose`

  变更内容：变更了初始化参数策略。weight_init默认值由"normal"改为None，bias_init默认值由"zeros"改为None。

  说明：权重默认初始化方法由使用"normal"改为在内部使用HeUniform初始化。偏差默认初始化方法由"zeros"改为在内部使用Uniform初始化。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.1接口 </td>
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

### Bug fixes

- [I6TKLW] 修复了昇腾平台上MobileNetV2网络性能劣化的问题。
- [I7CP5H] 修复了昇腾平台上ASR网络训练失败的问题。
- [I7I3EZ] 修复了由于Pillow 10.0.0版本变更枚举接口导致run_check()失败的问题。若在低版本MindSpore遇到，则安装10.0.0以下版本Pillow避免此问题。
- [I7IZ8K] 修复了assignsub接口在PyNative下的精度问题。
- [I7HGY0] 修复了函数式编程，在PyNative模式数据下沉场景，loss不收敛的问题。
- [I7J4N3] 修复了Profiler动态Shape模式下生成Step Trace失败的问题。
- [I7J4N3] 修复了MindInsight并行策略视图展示暂无数据的问题。
- [I79YY4] 修复了PyNative模式下高阶微分时的SiLU算子错误。
- [I6NQJQ] 修复了PyNative模式下ScatterUpdate算子动态shape场景下执行概率性失败的问题。
- [I6Y4G5] 修复了Graph模式下Conv3D算子动态Shape场景下执行失败的问题。

### 贡献者

感谢以下人员做出的贡献:

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.1.0 Release Notes

### 主要特性和增强

#### MindSpore Lite云侧推理

- [STABLE] 支持Ascend硬件后端单卡大模型以及单机多卡分布式大模型高性能推理。
- [STABLE] Python API Ascend后端支持多模型共享工作空间（Workspace）内存。
- [STABLE] [通过ModelGroup新增支持多模型共享权重](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/cloud_infer/runtime_cpp.html#%E5%A4%9A%E6%A8%A1%E5%9E%8B%E5%85%B1%E4%BA%AB%E6%9D%83%E9%87%8D)，比如大模型场景下全量模型和增量模型共享权重。

#### API

新增ModelGroup [Python](https://www.mindspore.cn/lite/api/zh-CN/r2.1/mindspore_lite/mindspore_lite.ModelGroup.html#mindspore_lite.ModelGroup)和[C++](https://mindspore.cn/lite/api/zh-CN/r2.1/api_cpp/mindspore.html#modelgroup)接口，接口定义如下：

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

### 主要特性和增强

#### PyNative

- [Stable] 全面支持动态shape，算子支持度详见[nn接口动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/r2.0/note/dynamic_shape_nn.html)、[ops接口动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/r2.0/note/dynamic_shape_func.html)和[算子动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/r2.0/note/dynamic_shape_primitive.html)。

#### AutoParallel

- [STABLE] 新建MindFormers独立仓，提供分布式并行套件功能，替代mindspore.nn.transformer模块。
- [DEMO] 分布式Gather算子支持BatchDim属性。
- [DEMO] 流水线并行支持指定输入数据任意维度作为Batch维。

### API变更

#### 算子

- `mindspore.ops.AdaptiveAvgPool2D` 新增算子原语。
- `mindspore.ops.BatchToSpaceNDV2` 新增算子原语。
- `mindspore.ops.CeLU` 新增算子原语。
- `mindspore.ops.ExtractVolumePatches` 新增算子原语。
- `mindspore.ops.FFTWithSize` 新增算子原语。
- `mindspore.ops.FillDiagonal` 新增算子原语。
- `mindspore.ops.FractionalMaxPool3DWithFixedKsize` 新增算子原语。
- `mindspore.ops.Im2Col` 新增算子原语。
- `mindspore.ops.MaskedScatter` 新增算子原语。
- `mindspore.ops.MatrixBandPart` 新增算子原语。
- `mindspore.ops.MatrixInverse` 新增算子原语。
- `mindspore.ops.MaxPoolWithArgmaxV2` 新增算子原语。
- `mindspore.ops.Ormqr` 新增算子原语。
- `mindspore.ops.RandpermV2` 新增算子原语。
- `mindspore.ops.ResizeBicubic` 新增算子原语。
- `mindspore.ops.Triu` 新增算子原语。
- `mindspore.ops.Zeta` 新增算子原语。

#### 非兼容性接口变更

- 接口名称：mindspore.ops.MultitypeFuncGraph

  变更内容：该接口参数doc_url在MindSpore 2.0.0.rc1版本作为测试特性，MindSpore 2.0.0版本优化后用户不需要额外配置此参数，故此参数在MindSpore 2.0.0版本删除。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0 接口</td>
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

- 接口名称：mindspore.set_context(auto_tune_mode="GA,RL")

  变更内容：下线算子AutoTune调优工具，删除auto_tune_mode选项，未来会规划新的调优工具。

- 接口名称：mindspore.set_context(mode=PYNATIVE_MODE)

  变更内容：默认由GRAPH_MODE改为PYNATIVE_MODE。

  说明：原有使用方式若未设置运行模式，该变更会影响性能，需要额外设置图模式，则使用以下方式：
  mindspore.set_context(mode=GRAPH_MODE)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.train.Model.train

  变更内容：dataset_sink_mode 默认值由True改为False。

  说明：原有使用方式若未设置dataset_sink_mode，该变更会影响性能，需要额外设置数据下沉运行模式，则使用以下方式：
  Model.train(dataset_sink_mode=True)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.export

  变更内容：参数file_format由"AIR"改为不指定默认值。

  说明：原有使用方式若未设置file_format，需要额外设置file_format，则使用以下方式：
  mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.norm

  变更内容：扩展ord参数功能，支持多种形式。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, [0, 1], p=2)
  </pre></td>
  <td><pre>
  ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, ord=2, dim=(0, 1))
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.norm

  变更内容：扩展ord参数功能，支持多种形式。

  说明：参考ops.norm例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.dropout

  变更内容：删除seed0、seed1参数，新增参数seed=None。由返回Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout(x, p=0.5, seed0=0, seed1=0)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output, mask = dropout(x, p=0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout(input, p=0.5, training=True, seed=None)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output = ops.dropout(input, p=0.5，training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout2d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td>
  <pre>
  ops.dropout2d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout2d(input, 0.5)
  </pre>
  </td>
  <td>
  <pre>
  ops.dropout2d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout2d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout3d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout3d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout3d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout3d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout3d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.std

  变更内容：接口重构，接口使用方式更符合用户使用习惯。

  说明：原有unbiased如果已显示设置，采用以下替代方案：
  ddof=0替代unbiased=False，ddof=1替代unbiased=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.load_param_into_net

  变更内容：新增ckpt中未加载的参数作为返回值。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.nn.BCELoss

  变更内容：`reduction` 默认值由'none'变为'mean'。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  BCELoss(weight=None, reduction='none')
  >>> # 举例:
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
  >>> # 举例:
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

- 接口名称：mindspore.ops.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整第2个和第3个参数的顺序，修改并扩展split_size_or_sections功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.split(input_x, axis=0, output_num=1)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, axis=1, output_num=4)
  </pre>
  </td>
  <td><pre>
  ops.split(tensor, split_size_or_sections, axis=0)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, split_size_or_sections=1, axis=1)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整两个参数的位置，修改并扩展split_size_or_sections功能。

  说明：参考ops.split例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.pad

  变更内容：修改参数名paddings为padding，添加mode和value功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.pad(input_x, paddings)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = ((1, 2), (2, 1))
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  <td><pre>
  ops.pad(input_x, padding, mode='constant', value=None)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = (2, 1, 1, 2)
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.meshgrid

  变更内容：入参由inputs改为*input。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.meshgrid(inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid((x, y, z), indexing='xy')
  </pre>
  </td>
  <td><pre>
  ops.meshgrid(*inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid(x, y, z, indexing='xy')
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.max

  变更内容：返回值调换顺序，由：“下标，最大值”改为“最大值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.max(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.max(input)
  >>> print(index, output)
  >>> 3 0.7
  </pre>
  </td>
  <td><pre>
  ops.max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.max(input, axis=0)
  >>> print(output, index)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.min

  变更内容：返回值调换顺序，由：“下标，最小值”改为“最小值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.min(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.min(input)
  >>> 0 0.0
  </pre>
  </td>
  <td><pre>
  ops.min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.min(input, keepdims=True)
  >>> 0.0 0
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.random_gamma

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.standard_laplace

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.standard_normal

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.bernoulli

  变更内容：seed的默认值由-1改为None。符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.data_sink

  变更内容：删除steps参数，jit参数名称修改为jit_config，新增input_signature参数。增加易用性，符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.conv2d

  变更内容：扩展接口功能，添加bias参数，修改参数名及参数顺序。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.vision.Pad

  变更内容：调整Pad、RandomCrop、RandomCropWithBbox入参padding，当Padding输入长度为2的序列时，行为将从使用第一个值填充左/上边界，使用第二个值填充右/下边界，变为使用第一个值填充左/右边界，使用第二个值填充上/下边界。

  说明：仅使用size为2的padding参数无法兼容旧版本的效果，需显式表示（左、右、上、下）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </pre>
  </td>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2,1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.Dataset.map

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：将batch方法拆分为：batch和padded_batch两个方法。pad_info参数从batch方法移动到padded_batch方法。

  说明：如需使用pad_info参数，改用padded_batch方法。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- [I62I3J] 修复bgcf网络在昇腾310上推理失败的问题
- [I7C2W3] 修复Pipeline并行场景下多loss打印编译失败问题

### 贡献者

感谢以下人员做出的贡献:

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

欢迎以任何形式对项目提供贡献！

## MindSpore 2.0.0-rc1 Release Notes

### 主要特性和增强

#### FrontEnd

- [BETA] 静态图模式下，函数及类方法支持"return None"、"return"、无"return"语法。
- [BETA] 静态图模式下，支持返回list类型对象。
- [BETA] 静态图模式下，变量条件时，支持"raise"语法。
- [STABLE] 函数式调用支持数据下沉模式。
- [BETA] nn下新增Transformer层，提供更加易用的Transformer API，无需定义batch_size，支持动态seq_length。

#### DataSet

- [STABLE] Ascend环境下，数据下沉模式超时等待时间调整，默认调整到1900s，以解决数据下沉模式时因环境资源竞争、计算量大等因素容易导致GetNext算子等待超时的问题。
- [STABLE] MindRecord提供Schema、样本数查询接口，并提供多进程并行写入功能，允许用户更快生成MindRecord数据文件。
- [STABLE] Dataset流水线支持处理任意Python对象，用法参考[数据pipeline支持Python对象](https://www.mindspore.cn/tutorials/zh-CN/r2.0/advanced/dataset/python_objects.html)。

#### AutoParallel

- [STABLE] 策略保存时支持保存完整策略。
- [STABLE] 支持Conv3D/MaxPool3D/AvgPool3D分布式算子。
- [STABLE] 支持PyNative+shard算子级并行+优化器并行：并行表达和Model进行解耦，提供基础的并行表达能力。
- [STABLE] 支持图模式算子级并行+优化器并行：并行表达和Model进行解耦，提供基础的并行表达能力。
- [BETA] 支持自定义分布式图切分，提升分布式训练的灵活性。

#### Runtime

- [STABLE] 控制流支持子图下沉。
- [STABLE] 支持CUDA 11.6。
- [STABLE] 支持List/Tuple/Scalar类型算子的算子选择和执行，配套Python原生表达。
- [STABLE] 硬件不支持的算子自动选择CPU算子。
- [STABLE] 支持子图内部异构执行。

#### Ascend

- [STABLE] 支持CANN溢出检测新方案和HCCL运行态溢出检测。
- [STABLE] 支持集合通信算子dump功能。

#### Profiler

- [STABLE] 丰富Profiler采集项配置，用户可以更细度地采集性能数据。

#### Dump

- [BETA] 单卡PyNatvie模式支持算子溢出检测。
- [BETA] Graph模式支持hccl算子dump。

### API变更

- [STABLE] 新增计算类API，如：MaxUnpool、ReplicationPad、GaussianNLLLoss等。
  详情请参考：<https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.html>。
- [STABLE] 扩展存量API功能，如：AvgPool、pad、norm、interplate等。

#### 算子

- [BETA] `mindspore.ops.AdaptiveAvgPool3D` 新增算子原语。
- [BETA] `mindspore.ops.AffineGrid` 新增算子原语。
- [BETA] `mindspore.ops.Angle` 新增算子原语。
- [BETA] `mindspore.ops.BartlettWindow` 新增算子原语。
- [BETA] `mindspore.ops.Bernoulli` 新增算子原语。
- [BETA] `mindspore.ops.BesselI0` 新增算子原语。
- [BETA] `mindspore.ops.BesselI1` 新增算子原语。
- [BETA] `mindspore.ops.BesselJ0` 新增算子原语。
- [BETA] `mindspore.ops.BesselJ1` 新增算子原语。
- [BETA] `mindspore.ops.BesselK0` 新增算子原语。
- [BETA] `mindspore.ops.BesselK0e` 新增算子原语。
- [BETA] `mindspore.ops.BesselK1` 新增算子原语。
- [BETA] `mindspore.ops.BesselK1e` 新增算子原语。
- [BETA] `mindspore.ops.BesselY0` 新增算子原语。
- [BETA] `mindspore.ops.BesselY1` 新增算子原语。
- [BETA] `mindspore.ops.Bincount` 新增算子原语。
- [BETA] `mindspore.ops.BlackmanWindow` 新增算子原语。
- [BETA] `mindspore.ops.ChannelShuffle` 新增算子原语。
- [BETA] `mindspore.ops.Cholesky` 新增算子原语。
- [BETA] `mindspore.ops.Col2Im` 新增算子原语。
- [BETA] `mindspore.ops.Complex` 新增算子原语。
- [BETA] `mindspore.ops.ComplexAbs` 新增算子原语。
- [BETA] `mindspore.ops.Cross` 新增算子原语。
- [BETA] `mindspore.ops.CTCLossV2` 新增算子原语。
- [BETA] `mindspore.ops.Cummin` 新增算子原语。
- [BETA] `mindspore.ops.Diag` 新增算子原语。
- [BETA] `mindspore.ops.Digamma` 新增算子原语。
- [BETA] `mindspore.ops.Expand` 新增算子原语。
- [BETA] `mindspore.ops.Fmax` 新增算子原语。
- [BETA] `mindspore.ops.Gcd` 新增算子原语。
- [BETA] `mindspore.ops.Geqrf` 新增算子原语。
- [BETA] `mindspore.ops.GLU` 新增算子原语。
- [BETA] `mindspore.ops.GridSampler2D` 新增算子原语。
- [BETA] `mindspore.ops.GridSampler3D` 新增算子原语。
- [BETA] `mindspore.ops.HammingWindow` 新增算子原语。
- [BETA] `mindspore.ops.Heaviside` 新增算子原语。
- [BETA] `mindspore.ops.Hypot` 新增算子原语。
- [BETA] `mindspore.ops.Igamma` 新增算子原语。
- [BETA] `mindspore.ops.IndexFill` 新增算子原语。
- [BETA] `mindspore.ops.InplaceIndexAdd` 新增算子原语。
- [BETA] `mindspore.ops.InplaceUpdateV2` 新增算子原语。
- [BETA] `mindspore.ops.Lcm` 新增算子原语。
- [BETA] `mindspore.ops.LeftShift` 新增算子原语。
- [BETA] `mindspore.ops.LogicalXor` 新增算子原语。
- [BETA] `mindspore.ops.Logit` 新增算子原语。
- [BETA] `mindspore.ops.LogSpace` 新增算子原语。
- [BETA] `mindspore.ops.LuUnpack` 新增算子原语。
- [BETA] `mindspore.ops.MatrixDiagPartV3` 新增算子原语。
- [BETA] `mindspore.ops.MatrixDiagV3` 新增算子原语。
- [BETA] `mindspore.ops.MatrixSetDiagV3` 新增算子原语。
- [BETA] `mindspore.ops.MaxPool3DWithArgmax` 新增算子原语。
- [BETA] `mindspore.ops.MaxUnpool2D` 新增算子原语。
- [BETA] `mindspore.ops.MaxUnpool3D` 新增算子原语。
- [BETA] `mindspore.ops.MultiMarginLoss` 新增算子原语。
- [BETA] `mindspore.ops.MultinomialWithReplacement` 新增算子原语。
- [BETA] `mindspore.ops.Mvlgamma` 新增算子原语。
- [BETA] `mindspore.ops.NanToNum` 新增算子原语。
- [BETA] `mindspore.ops.NextAfter` 新增算子原语。
- [BETA] `mindspore.ops.Orgqr` 新增算子原语。
- [BETA] `mindspore.ops.Polygamma` 新增算子原语。
- [BETA] `mindspore.ops.ResizeBilinearV2` 新增算子原语。
- [BETA] `mindspore.ops.RightShift` 新增算子原语。
- [BETA] `mindspore.ops.ScatterNdDiv` 新增算子原语。
- [BETA] `mindspore.ops.ScatterNdMul` 新增算子原语。
- [BETA] `mindspore.ops.SearchSorted` 新增算子原语。
- [BETA] `mindspore.ops.Sinc` 新增算子原语。
- [BETA] `mindspore.ops.Trace` 新增算子原语。
- [BETA] `mindspore.ops.Tril` 新增算子原语。
- [BETA] `mindspore.ops.TrilIndices` 新增算子原语。
- [BETA] `mindspore.ops.TriuIndices` 新增算子原语。
- [BETA] `mindspore.ops.UniqueConsecutive` 新增算子原语。
- [STABLE] `mindspore.ops.Cummax` 新增算子原语。
- [STABLE] `mindspore.ops.FillV2` 新增算子原语。
- [STABLE] `mindspore.ops.IsClose` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixSolve` 新增算子原语。
- [STABLE] `mindspore.ops.Median` 新增算子原语。
- [STABLE] `mindspore.ops.MultilabelMarginLoss` 新增算子原语。
- [STABLE] `mindspore.ops.NonZero` 新增算子原语。
- [STABLE] `mindspore.ops.Pdist` 新增算子原语。
- [STABLE] `mindspore.ops.Polar` 新增算子原语。
- [STABLE] `mindspore.ops.RandomGamma` 新增算子原语。
- [STABLE] `mindspore.ops.RandomPoisson` 新增算子原语。
- [STABLE] `mindspore.ops.RandomShuffle` 新增算子原语。
- [STABLE] `mindspore.ops.Renorm` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMax` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMin` 新增算子原语。
- [STABLE] `mindspore.ops.Svd` 新增算子原语。
- [STABLE] `mindspore.ops.TripletMarginLoss` 新增算子原语。

#### 删除接口

- `mindspore.compression`特性在MindSpore 1.8版本已经废弃，在当前版本被删除。用户可以使用[昇思金箍棒](https://gitee.com/mindspore/golden-stick)作为`mindspore.compression`的替代品来实现MindSpore中的量化感知训练算法。
- `mindspore.dataset.close_pool`、`mindspore.dataset.to_device`、`mindspore.dataset.set_dynamic_columns` 接口在之前版本已废弃，当前版本正式删除。

#### 非兼容性接口变更

- 接口名称：mindspore.set_context(mode=PYNATIVE_MODE)

  变更内容：默认由GRAPH_MODE改为PYNATIVE_MODE。

  说明：原有使用方式若未设置运行模式，该变更会影响性能，需要额外设置图模式，则使用以下方式：
  mindspore.set_context(mode=GRAPH_MODE)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.train.Model.train

  变更内容：dataset_sink_mode 默认值由True改为False。

  说明：原有使用方式若未设置dataset_sink_mode，该变更会影响性能，需要额外设置数据下沉运行模式，则使用以下方式：
  Model.train(dataset_sink_mode=True)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.export

  变更内容：参数file_format由"AIR"改为不指定默认值。

  说明：原有使用方式若未设置file_format，需要额外设置file_format，则使用以下方式：
  mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.norm

  变更内容：扩展ord参数功能，支持多种形式。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, [0, 1], p=2)
  </pre>
  </td>
  <td><pre>
  ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                          [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, ord=2, dim=(0, 1))
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.norm

  变更内容：扩展ord参数功能，支持多种形式。

  说明：参考ops.norm例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.dropout

  变更内容：删除seed0、seed1参数，新增参数seed=None。由返回Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td>
  <pre>
  ops.dropout(x, p=0.5, seed0=0, seed1=0)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output, mask = dropout(x, p=0.5)
  </pre>
  </td>
  <td>
  <pre>
  ops.dropout(input, p=0.5, training=True, seed=None)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output = ops.dropout(input, p=0.5，training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout2d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td>
  <pre>
  ops.dropout2d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout2d(input, 0.5)
  </pre>
  </td>
  <td>
  <pre>
  ops.dropout2d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout2d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout3d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.dropout3d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout3d(input, 0.5)
  </pre>
  </td>
  <td><pre>
  ops.dropout3d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout3d(input, 0.5, training=True)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.std

  变更内容：接口重构，接口使用方式更符合用户使用习惯。

  说明：原有unbiased如果已显示设置，采用以下替代方案：
  ddof=0替代unbiased=False，ddof=1替代unbiased=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.load_param_into_net

  变更内容：新增ckpt中未加载的参数作为返回值。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.nn.BCELoss

  变更内容：`reduction` 默认值由'none'变为'mean'。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  BCELoss(weight=None, reduction='none')
  >>> # 举例:
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
  >>> # 举例:
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

- 接口名称：mindspore.ops.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整第2个和第3个参数的顺序，修改并扩展split_size_or_sections功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.split(input_x, axis=0, output_num=1)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, axis=1, output_num=4)
  </pre>
  </td>
  <td><pre>
  ops.split(tensor, split_size_or_sections, axis=0)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, split_size_or_sections=1, axis=1)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整两个参数的位置，修改并扩展split_size_or_sections功能。

  说明：参考ops.split例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.pad

  变更内容：修改参数名paddings为padding，添加mode和value功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.pad(input_x, paddings)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = ((1, 2), (2, 1))
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  <td><pre>
  ops.pad(input_x, padding, mode='constant', value=None)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = (2, 1, 1, 2)
  >>> output = ops.pad(input_x, paddings)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.meshgrid

  变更内容：入参由inputs改为*input。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.meshgrid(inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid((x, y, z), indexing='xy')
  </pre>
  </td>
  <td><pre>
  ops.meshgrid(*inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid(x, y, z, indexing='xy')
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.max

  变更内容：返回值调换顺序，由：“下标，最大值”改为“最大值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.max(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.max(input)
  >>> print(index, output)
  >>> 3 0.7
  </pre>
  </td>
  <td><pre>
  ops.max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.max(input, axis=0)
  >>> print(output, index)
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.min

  变更内容：返回值调换顺序，由：“下标，最小值”改为“最小值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  ops.min(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.min(input)
  >>> 0 0.0
  </pre>
  </td>
  <td><pre>
  ops.min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.min(input, keepdims=True)
  >>> 0.0 0
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.random_gamma

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.standard_laplace

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.standard_normal

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.bernoulli

  变更内容：seed的默认值由-1改为None。符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.data_sink

  变更内容：删除steps参数，jit参数名称修改为jit_config，新增input_signature参数。增加易用性，符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.ops.conv2d

  变更内容：扩展接口功能，添加bias参数，修改参数名及参数顺序。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.vision.Pad

  变更内容：调整Pad、RandomCrop、RandomCropWithBbox入参padding，当Padding输入长度为2的序列时，行为将从使用第一个值填充左/上边界，使用第二个值填充右/下边界，变为使用第一个值填充左/右边界，使用第二个值填充上/下边界。

  说明：仅使用size为2的padding参数无法兼容旧版本的效果，需显式表示（左、右、上、下）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </pre>
  </td>
  <td><pre>
  mindspore.dataset.vision.Pad(padding=(1,2,1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.Dataset.map

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：将batch方法拆分为：batch和padded_batch两个方法。pad_info参数从batch方法移动到padded_batch方法。

  说明：如需使用pad_info参数，改用padded_batch方法。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
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

- [I66PE6] 修复 AssignSub算子异常入参导致core dump的问题。

- [I6F5E6] 修复 data_sink 方法在Ascend上执行超时的问题。

### 其它

- Windows系统支持由于还在优化中，rc版本暂不支持，将在2.0正式版本提供下载。

### 贡献者

感谢以下人员做出的贡献：

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.0.0-rc1 Release Notes

### 主要特性和增强

#### MindSpore Lite云侧推理

原MindSpore Lite版本主要面向手机、车机等边缘设备，新增云侧推理版本支持云侧多后端硬件资源的场景，支持Ascend及Nvidia GPU推理专用卡，高效利用云侧多核资源。

原通过MindSpore训练版本集成的推理方式可以变更为基于MindSpore Lite进行适配集成，具体可参考[云侧推理快速入门](https://mindspore.cn/lite/docs/zh-CN/r2.0/quick_start/one_hour_introduction_cloud.html)，如果想要保持原始集成方式可以参考[MindSpore推理FAQ](https://mindspore.cn/docs/zh-CN/r2.0/faq/inference.html)。

- [STABLE] 支持MindIR模型文件。
- [STABLE] 支持将第三方Onnx、Tensorflow、Caffe模型通过MindSpore Lite转换工具转换为MindIR模型文件。
- [STABLE] 一个发布包支持多种硬件后端：Ascend、Nvidia GPU、CPU。
- [STABLE] 支持`Model`接口和`ModelParallelRunner`并行推理接口。
- [STABLE] 支持C++、Python和Java推理接口。

#### API

- 因原Python API配置参数较多、使用较复杂，因此在2.0版本针对Python API易用性进行优化，包括类构造方法、类属性的调整等，此外2.0及之后的Python API将整合到云侧推理场景，与旧版本不兼容。详细参见[Python API说明文档](https://www.mindspore.cn/lite/api/zh-CN/r2.0/mindspore_lite.html)。

## MindSpore 2.0.0-alpha Release Notes

### 主要特性和增强

#### PyNative

- MindSpore默认模式切换成PyNative模式。需要手动设置模式可以参考文档[计算图](https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/advanced/compute_graph.html)。
- 完成动态shape执行方案重构，提升反向构图性能，支持非padding方案的动态shape网络编程，当前主要验证网络Transformer-GPU、YOLOV5-GPU、ASR-Ascend。从[models仓](https://gitee.com/mindspore/models/tree/dynamic_shape)获取Transformer-GPU和YOLOV5-GPU。Ascend后端受算子适配度限制，只支持下列算子：Add、Assign、BatchMatMul、BiasAdd、BiasAddGrad、Cast、Conv2D、Conv2DBackpropFilter、Conv2DBackpropInput、CTCLoss、Div、Dropout、DropoutDoMask、Equal、ExpandDims、Gather、GetNext、LayerNorm、LayerNormGrad、LessEqual、Load、Log、LogicalAnd、LogicalNot、LogicalOr、LogSoftmax、LogSoftmaxGrad、MatMul、Maximum、Mul、Neg、NotEqual、NPUAllocFloatStatus、NPUClearFloatStatus、OneHot、RealDiv、Reciprocal、ReduceMean、ReduceSum、ReLU、ReluGrad、Reshape、Select、Softmax、StridedSlice、Sub、Tile、Transpose、UnsortedSegmentSum、ZerosLike。其余算子未经过完整验证，请酌情使用。

#### DataSet

- TFRecordDataset API支持直接读取通过GZIP或ZLIB压缩后的TFRecord文件。
- NumpySlicesDataset API支持同时处理不同维度的数据。
- 优化错误日志信息的结构，展示更清晰的调用栈信息便于调试、定位问题。
- 修复分布式训练场景下 `mindspore.dataset.config.set_seed` 对随机种子设置不生效的问题。

#### AutoParallel

- 支持更多算子分布式能力。

  Element Wise类算子：AddN、 BitwiseAnd、 BitwiseOr、 BitwiseXor、 CumProd、 HShrink、 HSigmoid、 IsFinite、 Mish、 MulNoNan、 Rint、 SeLU、 SoftShrink、 TruncateDiv、 TruncateMod、 Xdivy Xlogy、 InplaceAdd、 InplacSub、 InplaceUpdate、 Cdist、 L2Loss、 Lerp。

  Math类算子：SquaredDifference、 Erfinv、 MaskedFill、 SplitV、 Gamma、 KLDivLoss、 LinSpace。Scatter类算子：ScatterAdd、ScatterDiv、ScatterMax、ScatterMul、ScatterNdAdd、ScatterNdSub、ScatterNdUpdate、ScatterSub、TensorScatterAdd、TensorScatterDiv、TensorScatterMax、TensorScatterMax、TensorScatterMul、TensorScatterAdd、TensorScatterUpdate。

- 增加`transform_checkpoints`和`transform_checkpoint_by_rank`接口。给定转换前后的策略文件，即可实现对分布式权重转换。详情请参考[分布式弹性训练与推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/parallel/resilience_train_and_predict.html)。

### API变更

#### 算子

- [STABLE] `mindspore.ops.AdaptiveMaxPool3D` 新增算子原语。
- [STABLE] `mindspore.ops.AdjustHue` 新增算子原语。
- [STABLE] `mindspore.ops.BartlettWindow` 新增算子原语。
- [STABLE] `mindspore.ops.BesselJ0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselJ1` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK0e` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK1` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK1e` 新增算子原语。
- [STABLE] `mindspore.ops.BesselY0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselY1` 新增算子原语。
- [STABLE] `mindspore.ops.Betainc` 新增算子原语。
- [STABLE] `mindspore.ops.Bincount` 新增算子原语。
- [STABLE] `mindspore.ops.BlackmanWindow` 新增算子原语。
- [STABLE] `mindspore.ops.Bucketize` 新增算子原语。
- [STABLE] `mindspore.ops.CombinedNonMaxSuppression` 新增算子原语。
- [STABLE] `mindspore.ops.CompareAndBitpack` 新增算子原语。
- [STABLE] `mindspore.ops.Complex` 新增算子原语。
- [STABLE] `mindspore.ops.DataFormatVecPermute` 新增算子原语。
- [STABLE] `mindspore.ops.EuclideanNorm` 新增算子原语。
- [STABLE] `mindspore.ops.Expand` 新增算子原语。
- [STABLE] `mindspore.ops.ExtractGlimpse` 新增算子原语。
- [STABLE] `mindspore.ops.FillDiagonal` 新增算子原语。
- [STABLE] `mindspore.ops.FractionalAvgPool` 新增算子原语。
- [STABLE] `mindspore.ops.FractionalMaxPool` 新增算子原语。
- [STABLE] `mindspore.ops.Gcd` 新增算子原语。
- [STABLE] `mindspore.ops.HammingWindow` 新增算子原语。
- [STABLE] `mindspore.ops.Histogram` 新增算子原语。
- [STABLE] `mindspore.ops.HSVToRGB` 新增算子原语。
- [STABLE] `mindspore.ops.Lcm` 新增算子原语。
- [STABLE] `mindspore.ops.LeftShift` 新增算子原语。
- [STABLE] `mindspore.ops.ListDiff` 新增算子原语。
- [STABLE] `mindspore.ops.LogSpace` 新增算子原语。
- [STABLE] `mindspore.ops.Lstsq` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixDiagPartV3` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixDiagV3` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixExp` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixPower` 新增算子原语。
- [STABLE] `mindspore.ops.MaxPool3DWithArgmax` 新增算子原语。
- [STABLE] `mindspore.ops.MaxUnpool2D` 新增算子原语。
- [STABLE] `mindspore.ops.MultilabelMarginLoss` 新增算子原语。
- [STABLE] `mindspore.ops.NextAfter` 新增算子原语。
- [STABLE] `mindspore.ops.Orgqr` 新增算子原语。
- [STABLE] `mindspore.ops.ReduceStd` 新增算子原语。
- [STABLE] `mindspore.ops.RGBToHSV` 新增算子原语。
- [STABLE] `mindspore.ops.RightShift` 新增算子原语。
- [STABLE] `mindspore.ops.SampleDistortedBoundingBoxV2` 新增算子原语。
- [STABLE] `mindspore.ops.ScaleAndTranslate` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterAddWithAxis` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdDiv` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMax` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMul` 新增算子原语。
- [STABLE] `mindspore.ops.STFT` 新增算子原语。
- [STABLE] `mindspore.ops.Trace` 新增算子原语。
- [STABLE] `mindspore.ops.UpsampleNearest3D` 新增算子原语。
- [STABLE] `mindspore.ops.UpsampleTrilinear3D` 新增算子原语。
- [STABLE] `mindspore.parallel.transform_checkpoints` 新增分布式权重转换接口。
- [STABLE] `mindspore.parallel.transform_checkpoint_by_rank` 新增分布式权重转换接口。

#### 非兼容性变更

##### Python API

- `mindspore.ms_function`接口名替换为`mindspore.jit`，`mindspore.ms_function` 将在未来版本中弃用并删除。
- `mindspore.ms_class`接口名替换为`mindspore.jit_class`，`mindspore.ms_class` 将在未来版本中弃用并删除。
- `mindspore.ops.ms_kernel`接口名替换为`mindspore.ops.kernel`，`mindspore.ops.ms_kernel` 将在未来版本中弃用并删除。
- `mindspore.dataset.map`接口参数 `column_order` 不再生效，使用`mindspore.dataset.project`替换。
- `mindspore.dataset.close_pool`、`mindspore.dataset.to_device`、`mindspore.dataset.set_dynamic_columns` 接口在之前版本已废弃，当前版本正式删除。

### Bug fixes

- 修复混合精度函数式接口在图模式下不能修改后端驱动的问题。
- 修复以下网络在单P场景下用户可自动传入device_id（mobilenetv1/fasterrcnn/yolov3/yolov4/yolov5/unet/openpose/simplepose/crnn/gnmtv2/faceattribute/facequality/facedetection） 。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.10.1 Release Notes

### 问题修复

- 修复logsumexp防溢出处理中未考虑指定axis的问题
- 修复proto文件的编译依赖问题
- 修复print算子打印结果不正常的问题
- 修复equal算子越界问题
- 修复函数被@jit修饰后，导致的cell_id解析不正确的问题
- 修复GNN场景数据类型校验错误
- 修复Dataset map多进程退化成线程的问题

### 贡献者

感谢以下人员做出的贡献:

archer2049, caifubi, chenfei_mindspore, gaoshuanglong, Greatpan, guozhijian, huoxinyou, Kxiong, lanzhineng, lijunbin, liubuyu, liuchuting, luochao60, lyqlola, nomindcarry, TuDouNi, xiaotianci, xupan, yangshuo, yefeng, YingtongHu, yuchaojie, zhoufeng, ZPaC, 刘勇琪, 吕昱峰, 王禹程, 于振华.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.10.0 Release Notes

### 主要特性和增强

#### DataSet

- [STABLE]下沉模式超时等待时间调整，默认调整到600s，以解决数据下沉模式时因环境资源竞争、计算量大等因素容易导致GetNext算子等待超时的问题。

### Bug fixes

- 修复AMP中部分Primitive算子无法在图模式下实例化导致接口不可用的问题。
- 修复昇腾平台算力切分场景下LSTM网络中DynamicRNN算子执行失败的问题。
- 修复mobilenet, fasterrcnn, yolo等网络单卡训练脚本DEVICE_ID在启动脚本中写死的问题。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.10.0 Release Notes

### Bug fixes

- 修复Arithmetic类CPU算子动态shape场景下可能的计算精度问题。
- 修复Deconv int8量化算子重量化写入地址错误问题。

## MindSpore 1.9.0 Release Notes

### 主要特性和增强

#### FrontEnd

- [STABLE] 新增面向对象+函数式融合编程范式，提供 `mindspore.amp.LossScaler` 、 `mindspore.amp.DynamicLossScaler` 、 `mindspore.amp.StaticLossScaler` 、 `mindspore.amp.auto_mixed_precision` 、 `mindspore.amp.all_finite` 等融合编程范式下的混合精度接口。

### API变更

#### 算子

- [STABLE] `nn.AdaptiveAvgPool3d` 新增nn接口。
- [STABLE] `ops.adaptive_avg_pool3d` 新增functional接口。
- [STABLE] `ops.addcdiv` 新增functional接口。
- [STABLE] `ops.addcmul` 新增functional接口。
- [STABLE] `ops.approximate_equal` 新增GPU、CPU支持。
- [STABLE] `ops.atanh` 新增GPU支持。
- [STABLE] `ops.bessel_i0` 新增GPU支持。
- [STABLE] `ops.bessel_i0e` 新增Ascend支持。
- [STABLE] `ops.bessel_i1` 新增GPU支持。
- [STABLE] `ops.bessel_i1e` 新增Ascend、GPU支持。
- [STABLE] `ops.bessel_j0` 新增GPU支持。
- [STABLE] `ops.bessel_j1` 新增GPU支持。
- [STABLE] `ops.bessel_k0` 新增GPU支持。
- [STABLE] `ops.bessel_k0e` 新增GPU支持。
- [STABLE] `ops.bessel_k1` 新增GPU支持。
- [STABLE] `ops.bessel_k1e` 新增GPU支持。
- [STABLE] `ops.bessel_y0` 新增GPU支持。
- [STABLE] `ops.bessel_y1` 新增GPU支持。
- [STABLE] `ops.bias_add` 新增functional接口。
- [STABLE] `ops.bitwise_and` 新增GPU支持。
- [STABLE] `ops.bitwise_or` 新增GPU支持。
- [STABLE] `ops.bitwise_xor` 新增GPU支持。
- [STABLE] `ops.grid_sample` 新增Ascend支持。
- [STABLE] `ops.inplace_update` 新增CPU支持。
- [STABLE] `ops.isclose` 新增Ascend、GPU支持。
- [STABLE] `ops.isnan` 新增Ascend支持。
- [STABLE] `ops.lerp` 新增GPU支持。
- [STABLE] `ops.random_poisson` 新增functional接口。
- [STABLE] `ops.reverse_sequence` 新增functional接口。
- [STABLE] `ops.scatter_mul` 新增GPU支持。
- [STABLE] `ops.scatter_nd_max` 新增functional接口。
- [STABLE] `ops.scatter_nd_min` 新增functional接口。
- [STABLE] `ops.SparseToDense` 新增GPU支持。
- [STABLE] `ops.square` 新增functional接口。
- [STABLE] `ops.standard_laplace` 新增GPU支持。
- [STABLE] `ops.std` 新增functional接口。
- [STABLE] `ops.trunc` 新增Ascend、GPU支持。
- [STABLE] `ops.unsorted_segment_sum` 新增functional接口。
- [STABLE] `ops.xdivy` 新增functional接口。
- [STABLE] `ops.xlogy` 新增GPU支持。
- `ops.poisson` 接口废弃使用，对应新接口为 `ops.random_poisson` 。
- `ops.SparseApplyAdagrad` 接口废弃使用，可使用 `ops.SparseApplyAdagradV2` 接口替代。

### Bug fixes

- [BUGFIX] 修改混合精度O2 level的判断逻辑，在原来屏蔽 `BatchNorm1d` 、 `BatchNorm2d` 算子的基础上，添加另外两个屏蔽算子`BatchNorm3d`和`LayerNorm`，这4个算子依然用float32数据类型计算。

- [BUGFIX] Dataset处理字符串类型数据时，若调用`create_dict_iterator`或`create_tuple_iterator`接口时指定了`output_numpy=True`，获取到的数据会是`numpy.bytes_`类型。修复此问题后接口会直接返回`numpy.str_`类型数据，用户无需再对其进行字符串解码操作。同样，在使用自定义数据处理函数时，接收到的数据也将直接是`numpy.str_`类型，与原始数据类型相匹配。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, liyanliu, lizhenyu, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, panfengfeng, panyifeng, Payne, peixu_ren, Pengyongrong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanyuan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.8.1 Release Notes

### API变更

#### 算子

- [STABLE] ops.ApplyAdagradDA 新增GPU、CPU支持。
- [STABLE] ops.ApplyAdagradV2 新增CPU支持。
- [STABLE] ops.ApplyCenteredRmsProp 新增Ascend动态shape支持。
- [STABLE] ops.ApplyFtrl 新增CPU支持。
- [STABLE] ops.ApplyGradientDescent 新增CPU支持。
- [STABLE] ops.ApplyPowerSign 新增CPU支持。
- [STABLE] ops.ApplyProximalAdagrad 新增GPU、CPU支持。
- [STABLE] ops.ApplyRmsProp 新增Ascend动态shape支持。
- [STABLE] ops.max 新增functional接口。
- [STABLE] ops.atan2 新增functional接口。
- [STABLE] ops.cummax 新增GPU支持。
- [STABLE] ops.cummin 新增GPU、CPU支持。
- [STABLE] ops.diag 新增GPU支持。
- [STABLE] ops.expand_dims 新增functional接口。
- [STABLE] ops.gather_elements 新增functional接口。
- [STABLE] ops.grid_sample 新增GPU支持。
- [STABLE] ops.hardswish 新增Ascend支持。
- [BETA] ops.index_fill 新增GPU支持。
- [BETA] ops.inplace_update 新增CPU支持。
- [BETA] nn.InstanceNorm1d 新增GPU支持。
- [BETA] nn.InstanceNorm2d 新增GPU支持。
- [BETA] nn.InstanceNorm3d 新增GPU支持。
- [STABLE] ops.log1p 新增functional接口。
- [STABLE] ops.masked_fill 新增GPU、CPU支持。
- [BETA] ops.matrix_diag_part 新增GPU支持。
- [BETA] ops.matrix_diag 新增GPU支持。
- [BETA] ops.matrix_set_diag 新增GPU支持。
- [STABLE] ops.max_pool3d 新增GPU支持。
- [STABLE] ops.nll_loss 新增functional接口。
- [STABLE] ops.one_hot 新增functional接口。
- [STABLE] ops.pad 新增functional接口。
- [STABLE] ops.random_gamma 新增CPU支持。
- [STABLE] ops.amax 新增functional接口。
- [STABLE] ops.mean 新增functional接口。
- [STABLE] ops.amin 新增functional接口。
- [STABLE] ops.prod 新增functional接口。
- [STABLE] ops.renorm 新增Ascend、GPU、CPU支持。
- [BETA] ops.tensor_scatter_elements 新增Ascend、GPU、CPU支持。
- [STABLE] ops.scatter_max 新增GPU支持。
- [STABLE] ops.scatter_min 新增GPU支持。
- [STABLE] ops.scatter_nd 新增functional接口。
- [STABLE] ops.scatter_nd_max 新增GPU支持。
- [STABLE] ops.scatter_update 新增functional接口。
- [STABLE] ops.binary_cross_entropy_with_logits 新增CPU支持。
- [STABLE] ops.smooth_l1_loss 新增functional接口。
- [STABLE] ops.space_to_batch_nd 新增CPU支持。
- [STABLE] ops.SparseApplyAdagrad 新增GPU、CPU支持。
- [STABLE] ops.sparse_segment_mean 新增GPU、CPU支持。
- [STABLE] ops.squeeze 新增functional接口。
- [STABLE] ops.standard_laplace 新增CPU支持。
- [BETA] nn.ReflectionPad1d 新增Ascend、GPU、CPU支持。
- [BETA] nn.ReflectionPad2d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.SiLU 新增Ascend、GPU、CPU支持。
- [STABLE] ops.transpose 新增functional接口。
- [STABLE] ops.uniform_candidate_sampler 新增CPU支持。
- [STABLE] ops.uniform 新增functional接口。
- [STABLE] ops.unique_with_pad 新增GPU支持。
- [STABLE] ops.unstack 新增functional接口。
- [BETA] ops.interpolate 新增GPU、CPU支持。
- [STABLE] ops.xdivy 新增CPU支持。
- [STABLE] ops.xlogy 新增CPU支持。

## MindSpore 1.8.0 Release Notes

### 主要特性和增强

#### FrontEnd

- [BETA]  提供`mindspore.train.Model.fit` API，增加两种callback方法 `mindspore.train.callback.EarlyStopping` 和 `mindspore.train.callback.ReduceLROnPlateau`。
- [BETA] 自定义算子支持Julia算子。
- [BETA] 自定义算子支持Hybrid DSL算子。
- [STABLE] export()接口支持自定义加密算法导出模型，load()接口支持自定义解密算法导入模型。
- [BETA]   [动静统一] [易用性] 图编译支持常量类型设置可变(1.8版本支持tuple/list/dict)。
- [BETA]   [动静统一] 常量场景下控制流内支持JIT Fallback功能。
- [STABLE] [动静统一] 支持图模式常量场景下Python raise语句。
- [STABLE] [动静统一] 支持图模式常量场景下Python assert语句。
- [STABLE] [动静统一] 支持图模式常量场景下Python print语句。
- [STABLE] [动静统一] 支持图模式str.format()方法。
- [STABLE] [动静统一] 支持图模式用slice方法对list赋值。
- [STABLE] [动静统一] 图模式支持创建和调用自定义类的实例。
- [STABLE] [动静统一] 支持从Cell数组/自定义类数组中获取类的属性。
- [STABLE] [动静统一] 图模式下isinstance支持场景扩展。
- [STABLE] 自定义算子修饰符'ms_hybrid'重名为'ms_kernel'。
- [BETA] 自定义算子Hybrid DSL支持CPU后端。
- [BETA] 自定义算子昇腾后端新增自定义调度原语语法支持。

#### PyNative

- [STABLE] 实现AdamWeightDecay算子，替代原有小算子组合方式。
- [STABLE] 动态图下使用动静结合的方式执行优化器。
- [STABLE] 优化PyNative反向图和ms_function的执行性能。

#### Auto Parallel

- [STABLE] 对接AllToAll单算子模式。在图编译等级为O0下，支持AllToAll算子调用。
- [STABLE] 整图下沉支持MPI启动。整图下沉的模式下，支持使用MPI的方式启动。
- [STABLE] 模型权重的Seed提供并行接口配置。在用户不通过mindspore.set_seed设置随机数种子时，每个参数初始化的随机数种子为当前分片索引决定。当配置随机数种子之后，相同shape以及相同切分策略的权重，其初始化的结果一致。
- [STABLE] HCCL屏蔽内部全连接/非全连接。允许一次训练过程中同时有全连接AllToAllv和分级AllToAllv。
- [BETA] CPU优化器融合。通过优化器跨参数融合，将多个优化器算子按数据类型融合成，带来性能提升。目前已在CPU AdamWeightDecay优化器上做过验证。用户可以通过网络cell类中的flatten_weights方法启用该功能。

#### Executor

- [STABLE] 开放南向芯片对接接口。
- [STABLE] 使用多Actor融合执行提升运行时的执行性能。
- [STABLE] NopOp算子(eg. Reshape)执行消除。
- [STABLE] Embedding Cache架构切换统一分布式运行时。
- [STABLE] Parameter Server训练切换统一分布式运行时。
- [STABLE] 支持CPU Parameter Server模式训练。

#### DataSet

- [STABLE] 对于数据集对象使用map操作时，同时num_parallel_workers>1并且python_multiprocessing=True时，进行了多进程的机制优化，使得数据通道与子进程一一映射，避免了过多的文件句柄占用，同时close_pool这个接口也被删除。
- [STABLE] 新增一批Vision、Text和Audio类数据增强操作。
- [STABLE] 修复数据集类的flat_map方法未将结果展平的错误。
- [STABLE] 统一数据集增强API的导入路径，提供更简单的使用方法，请参阅[最新的API用法](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.dataset.vision.html)。

### API变更

#### 算子

- [STABLE] ops.adaptive_avg_pool2d 新增GPU支持。
- [BETA] ops.adaptive_max_pool2d  新增Ascend、GPU、CPU支持。
- [BETA] ops.approximate_equal 新增CPU支持。
- [STABLE] ops.argmin 新增CPU支持。
- [BETA] ops.assign_sub 新增CPU支持。
- [STABLE] ops.bernoulli 新增GPU支持。
- [BETA] ops.bessel_i0 新增CPU支持。
- [BETA] ops.bessel_i0e 新增CPU支持。
- [BETA] ops.bessel_i1 新增CPU支持。
- [BETA] ops.bessel_i1e 新增CPU支持。
- [STABLE] ops.bessel_j0 新增CPU支持。
- [STABLE] ops.bessel_j1 新增CPU支持。
- [STABLE] ops.bessel_k0 新增CPU支持。
- [STABLE] ops.bessel_k0e 新增CPU支持。
- [BETA] ops.bessel_k1 新增CPU支持。
- [BETA] ops.bessel_k1e 新增CPU支持。
- [STABLE] ops.bessel_y0 新增CPU支持。
- [STABLE] ops.bessel_y1 新增CPU支持。
- [STABLE] ops.bitwise_and 新增CPU支持。
- [STABLE] ops.bitwise_or 新增CPU支持。
- [STABLE] ops.bitwise_xor 新增CPU支持。
- [STABLE] ops.broadcast_to 新增functional接口。
- [BETA] ops.ceil 新增GPU、CPU支持。
- [BETA] ops.col2im 新增GPU支持。
- [BETA] ops.concat 新增functional接口。
- [STABLE] ops.cosh 新增GPU支持。
- [STABLE] ops.ctc_greedy_decoder 新增Ascend、CPU支持。
- [BETA] ops.DataFormatDimMap 新增GPU、CPU支持。
- [BETA] ops.dropout2d 新增GPU、CPU支持。
- [BETA] ops.dropout3d 新增CPU支持。
- [BETA] ops.erf 新增CPU支持。
- [BETA] ops.erfc 新增CPU支持。
- [STABLE] ops.expand_dims 新增functional接口。
- [STABLE] ops.fast_gelu 新增GPU、CPU支持。
- [STABLE] ops.flatten Ascend动态shape支持。
- [BETA] ops.ger 新增GPU、CPU支持。
- [STABLE] ops.gumbel_softmax 新增Ascend、GPU、CPU支持。
- [BETA] ops.hardshrink 新增GPU、CPU支持。
- [BETA] ops.index_add 新增CPU支持。
- [BETA] ops.inplace_add 新增CPU支持。
- [BETA] ops.inplace_sub 新增CPU支持。
- [STABLE] ops.intopk 新增CPU支持。
- [STABLE] ops.inv 新增GPU、CPU支持。
- [STABLE] ops.invert 新增GPU、CPU支持。
- [BETA] ops.isclose 新增CPU支持。
- [STABLE] ops.lerp 新增CPU支持。
- [BETA] ops.linspace 新增CPU支持。
- [BETA] ops.log_softmax 新增functional接口。
- [BETA] ops.norm 新增Ascend、GPU、CPU支持。
- [BETA] ops.lrn 新增CPU支持。
- [BETA] ops.masked_select 新增GPU支持。
- [BETA] ops.matrix_band_part 新增GPU、CPU支持。
- [BETA] ops.matrix_solve 新增GPU、CPU支持。
- [BETA] ops.meshgrid 新增CPU支持。
- [STABLE] ops.mish 新增CPU支持。
- [BETA] ops.nonzero  新增GPU支持。
- [STABLE] ops.padding 新增GPU、CPU支持。
- [BETA] ops.pow 新增Ascend动态shape支持。
- [BETA] ops.range 新增functional接口。
- [BETA] ops.round 新增Ascend动态shape支持。
- [STABLE] ops.scatter_add 新增Ascend动态shape支持。
- [STABLE] ops.scatter_div 新增Ascend动态shape支持。
- [BETA] ops.scatter_max 新增GPU支持。
- [BETA] ops.scatter_min 新增GPU支持。
- [BETA] ops.scatter_nd_add 新增CPU支持。
- [STABLE] ops.scatter_nd_div 新增GPU、CPU支持。
- [STABLE] ops.scatter_nd_min 新增GPU、CPU支持。
- [STABLE] ops.scatter_nd_mul 新增GPU、CPU支持。
- [BETA] ops.scatter_nd_sub 新增CPU支持。
- [STABLE] ops.scatter_update 新增Ascend动态shape支持。
- [BETA] ops.select 新增Ascend动态shape支持。
- [BETA] ops.selu 新增GPU、CPU支持。
- [BETA] ops.soft_shrink 新增GPU、CPU支持。
- [BETA] ops.softsign 新增CPU支持。
- [STABLE] ops.tan 新增GPU支持。
- [BETA] ops.tensor_scatter_add 新增Ascend、CPU支持。
- [STABLE] ops.tensor_scatter_div 新增GPU、CPU支持。
- [STABLE] ops.tensor_scatter_mul 新增GPU、CPU支持。
- [BETA] ops.tensor_scatter_sub 新增Ascend、CPU支持。
- [STABLE] nn.AdaptiveAvgPool1d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.AdaptiveMaxPool1d 新增Ascend、GPU、CPU支持。
- [BETA] nn.BiDense 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad1d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad2d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad3d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Hardtanh 新增Ascend、GPU、CPU支持。
- [STABLE] nn.HuberLoss 新增Ascend、GPU、CPU支持。
- [STABLE] nn.RReLU 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Tanhshrink 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Threshold 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ZeroPad2d 新增Ascend、GPU、CPU支持。
- [BETA] ops.unique_consecutive 新增GPU支持。
- [STABLE] ops.unsorted_segment_max 新增CPU支持。
- [STABLE] ops.unsorted_segment_min 新增CPU支持。
- [STABLE] ops.unsorted_segment_prod 新增GPU支持。

#### 非兼容性变更

##### Python API

- 不再支持DVPP模拟算法，删除 `mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg` 和 `mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg` 接口。
- LossMonitor中增加`on_train_epoch_end` 方法，实现在 `mindspore.train.Model.fit` 中使用时，打印epoch级别的metric信息。
- TimeMonitor打印内容变更，打印内容加入"train"或"eval"用于区分训练和推理阶段。
- load_checkpoint 接口的`filter_prefix`：不再支持空字符串("")，匹配规则由强匹配修改为模糊匹配。

#### import优化

mindspore.context、mindspore.parallel、mindspore.profiler、mindspore.train模块的接口可直接在mindspore模块使用。原有用法仍可以继续支持。

例如：

- `mindspore.context.set_context`可简化为`mindspore.set_context`。
- `mindspore.parallel.set_algo_parameters`可简化为`mindspore.set_algo_parameters`。
- `mindspore.profiler.Profiler`可简化为`mindspore.Profiler`。
- `mindspore.train.callback.Callback`可简化为`mindspore.train.Callback`。

API页面统一汇总至：<https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.html>。

### 贡献者

感谢以下人员做出的贡献：

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.8.0 Release Notes

### 主要特性和增强

#### API

- [STABLE] 新增模型转换的C++和Python API.
- [STABLE] 新增模型推理的Python API.

#### 后量化

- [STABLE] 后量化支持PerLayer量化，同时内置CLE算法优化精度。

## MindSpore 1.7.0 Release Notes

### 主要特性和增强

#### OS

- [STABLE] 支持Python 3.8版本（Linux/Windows/Mac）。
- [STABLE] 简化安装，提供详细安装指南和自动化安装脚本。
- [STABLE] Windows版本支持算子多线程。
- [STABLE] GCC兼容7.3到9.x版本。

#### FrontEnd

- [STABLE] 优化器支持动态权重衰减，即训练期间权重衰减值随着step的增加而变化。
- [STABLE] 增加四种创建Tensor的方法，分别是`mindspore.numpy.rand()`、`mindspore.numpy.randn()`、`mindspore.numpy.randint()`和`mindspore.ops.arange ()`。
- [STABLE] 增加一种callback方法 `mindspore.train.callback.History`。
- [BETA] 自定义算子支持Julia算子。
- [STABLE] 通过 `mindspore.ms_class` 类装饰器，支持获取用户自定义类的属性和方法。
- [STABLE] 支持同时存在副作用算子和控制流语句的网络的训练。
- [STABLE] 支持更复杂的控制流语法，比如在while的循环体里使用for语句。
- [STABLE] 通过减少子图数量，提升包含复杂控制流语法的网络的性能。

#### PyNative

- [STABLE] 在PyNative模式下支持hook函数功能，包括前向hook接口register_forward_pre_hook、register_forward_hook和反向hook接口register_backward_hook。
- [STABLE] 优化PyNative模式执行性能，并行执行前端Python与后端C++。

#### Auto Parallel

- [STABLE] 在MoE场景中支持TopK的路由、数据并行和优化器切分。
- [STABLE] 支持AllGather/ReduceScatter通信算子融合，在DATA_PARALLEL模式支持AllReduce按数据量大小编译。
- [STABLE] 在并行模式下支持ops.clip_by_global_norm。
- [STABLE] 在并行模式下支持AdaSum优化器。
- [STABLE] 支持自动优化器切分。
- [STABLE] 支持AlltoAll可配置开启，支持自动插入VirtualDatasetCell。
- [STABLE] 在流水线并行训练中，支持自动推断可训练的参数。
- [STABLE] 支持集群的设备数目不为2的幂次方。
- [STABLE] 在自动并行模式中支持策略传播。
- [STABLE] 在统一运行时中支持异构训练。
- [STABLE] 支持CPU的Adafactor算子。
- [STABLE] 支持Conv2d/Conv2D的H/W轴切分和Transpose算子。支持ResizeBilinear、ROIAlign、CropAndResize、BoundingBoxEncode、IOU和RandomChoiceWithMask等分布式算子。

#### Executor

- [BETA] [数据并行训练容灾](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/train_gpu.html#%E5%AE%B9%E7%81%BE%E6%81%A2%E5%A4%8D) 支持多卡数据并行训练容灾恢复。
- [BETA] 支持在CPU下的线程数搜索，获取最优线程数来执行。整个搜索过程需要耗时50个steps，整体的性能会在50个steps后达到稳定的状态。在测试性能的时候，需要以50个steps之后的数据作为标准。

#### DataSet

- [STABLE] 增加了数据处理API的差异文档，比较TensorFlow.data与MindSpore.dataset部分算子的差异，详见 [对比文档](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/tensorflow_api_mapping.html#tf-data)。
- [STABLE] Python多进程逻辑优化，保证不同异常场景的正常退出。
- [STABLE] 支持[自动数据加速](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/debug/dataset_autotune.html)，可以自适应调节数据处理管道的执行速度。
- [BETA] [数据处理异构加速](https://www.mindspore.cn/docs/zh-CN/r1.7/design/dataset_offload.html) 支持了新的数据增强操作: RandomColorAdjust、RandomSharpness和TypeCast。
- GeneratorDataset加载自定义数据集时，当`__getitem__/__next__`方法返回单个NumPy对象，对应会输出单个数据列。
- 用户在数据预处理中使用过多的进程数/线程数情况下，会出现错误RuntimeError: can't start new thread，可以通过 `ulimit -u 10240` 增加当前用户可用的线程/进程数解决。

### API变更

#### 非兼容性变更

##### Python API

- 修改register_backward_hook功能对应hook的梯度返回值类型，将梯度返回值统一改成tuple类型。([!31876](https://gitee.com/mindspore/mindspore/pulls/31876))
- 弃用的import用法： `import mindspore.dataset.engine.datasets as ds` ，因其import目录过深且过度依赖Python目录结构。推荐使用 `import mindspore.dataset as ds` ，更多参考详见 [API文档](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.dataset.html)。
- 新增`mindspore.ms_class` 接口，作为用户自定义类的类装饰器，使得MindSpore能够识别用户自定义类，并且获取这些类的属性和方法。([!30855](https://gitee.com/mindspore/mindspore/pulls/30855))
- `mindspore.SparseTensor`接口废弃使用，对应新接口为`mindspore.COOTensor`。 ([!28505](https://gitee.com/mindspore/mindspore/pulls/28505))
- Tensor新增一个入参`internal`，作为框架内部使用。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.7.0 Release Notes

### 主要特性和增强

#### 后量化

- [STABLE] 后量化支持动态量化算法。
- [BETA] 后量化模型支持在英伟达GPU上执行推理。
