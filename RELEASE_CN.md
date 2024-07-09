# MindSpore Release Notes

[View English](./RELEASE.md)

## MindSpore 2.2.15 Release Notes

### 问题修复

- [[#I9EYXX](https://gitee.com/mindspore/mindspore/issues/I9EYXX)] 修复编译缓存在msrun分布式并行启动方式下不使能的问题。
- [[#I86Z01](https://gitee.com/mindspore/mindspore/issues/I86Z01)] 修复ops.interpolate接口在PyNative模式下反向报错问题。
- [[#IA8OA3](https://gitee.com/mindspore/mindspore/issues/IA8OA3)] 修复万卡以上规模集群中偶现的host组网链接错误。
- [[#IABEYR](https://gitee.com/mindspore/mindspore/issues/IABEYR)] 修复万卡以上规模集群中偶现的errno: 110错误，并支持配置重试间隔时间。

### 贡献者

感谢以下人员做出的贡献:

xiaoyao, tianxiaodong, ZPaC

欢迎以任何形式对项目提供贡献！

## MindSpore 2.2.14 Release Notes

### 主要特性及增强

#### Parallel

- [STABLE] 将流水线并行的send/recv算子通信组改为world_group，避免创建冗余通信组，降低通信所需内存。
- [STABLE] 优化编译缓存，减少加载缓存时图转换过程，提升使用编译缓存性能。
- [BETA] 流水并行支持Interleave调度，优化micro batch开不大情况下的模型性能。
- [BETA] 优化流水线并行场景下模型转换速度，支持单个stage单独转换。

#### Profiler

- [BETA] 动态启停profiling，用户可以根据训练情况实时采集profiling数据，减少采集数据量。
- [BETA] Profiling通信算子耗时矩阵，用户通过分析通信算子耗时矩阵，找出集群通信性能瓶颈。

#### Dump

- [BETA] Dump保存的统计信息记录MD5值，用户可以通过MD5值确定张量值的微小差异。
- [BETA] Dump支持bfloat16数据类型，支撑用户定位bfloat16类型的算子精度问题。

### 问题修复

- [#I962EV] 修复了在CPU、GPU 环境上，使用cond接口输入为4d、5d、6d、7d、8d时，正向报错的问题。
- [#I96E5R] 修复Ascend平台PyNative场景下Mul算子输入为NCHW时报错的问题。
- [#I96I5D] 修复动态shape场景下，Scalar类型计算时输入类型错误的问题。
- [#I99QAB] 修复了部分场景下asnumpy接口无法正确识别bfloat16类型Tensor的问题。
- [#I9ADZS] 修复了故障恢复训练场景中，由于dataset恢复效率低导致网络训练出现数据超时的问题。
- [#I8Y9JT] 修复了nn.SGD优化器在loss_scale较大，weight_decay较小的部分特定场景下由于优化器执行顺序错误导致的部分网络训练不收敛的问题。

### 贡献者

感谢以下人员做出的贡献:

fary86, wanghenchang, haozhang, mengyuanli, emmmmtang, luoyang, zhupuxu, zhangyongxian, liuluobin, LLLRT, TuDouNi, hujiahui8, wangtongyu6, ligan, zhuguodong, yanghaoran, YingtongHu, liyejun, zjun, 徐永飞, chuht, 张树仁, 徐安越, DeshiChen, shenyaxin, liujunzhu, shunyuanhan, yuchaojie, yao_yf, 没有窗户的小巷, yeyunpeng2020, weiyang, KevinYi, hedongdong, zhouyaqiang0, Margaret_wangrui, zhanghaibo, moran, huangziling, 朱家兴, GuoZhibin, 李良灿, jiaxueyu, gaoyong10, Greatpan, 宦晓玲, melody, 俞涵, jiangshanfeng, XinDu, ling, caifubi, zhangyinxia, gengdongjie, Erpim, XianglongZeng, zhangminli, fengyixing, 冯一航, 黄勇, panzhihui, 胡彬, linqingke, wangshaocong

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.14 Release Notes

### 问题修复

- [I96PJC] 通过Mindspore Lite Python API加载ms格式的CLIP模型报错。

### 贡献者

感谢以下人员做出的贡献:

wangtongyu6, zhuguodong, 徐永飞, 徐安越, yeyunpeng2020, moran, XinDu, gengdongjie

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

- [#I8TPLY] 修复 SSD MobileNetV2 FPN 网络在Atlas 推理系列产品（配置 Ascend 310P AI 处理器）平台上的推理失败问题。

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

- [#I8ALW3]修复Faster R-CNN、DeepTextMask、RCNN-ResNet50等网络在Ascend 910上8卡训练RandomChoiceWithMask算子报错问题
- [#I8LKG7]修复UNet-2D在Ascend 910 1卡、8卡图编译报错问题
- [#I8KU3X]修复CRNN-ResNet34在Ascend 910 1卡、8卡PyNative模式下训练进程卡住问题
- [#I8KTHH]修复在Ascend 910 8卡上使能enable_parallel_optimizer=True，不使用allreduce分组融合时，BERT网络训练报错问题

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

- [STABLE] 支持用户可配置算子高精度/高性能模式，用户可以通过`context.set_context(ascend_config={"op_precision_mode": "/path/to/op_precision_config_file"})`对部分TBE算子配置高精度/高性能模式。
- [BETA] 支持用户可配置fp16进fp32出的算子，用户可以通过`context.set_context(ascend_config={"precision_mode": "force_fp32"})`对TBE Cube算子配置fp16进fp32出。
- [BETA] 去除jit level "O3"与GE流程强绑定，用户在执行GE流程时无需再设置`jit_level="O3"`。

#### Parallel

- [STABLE] 支持半自动/全自动模式下，非流水线并行场景的梯度累加特性，用户可以通过`net = GradAccumulationCell(net, micro_size)`方式，对网络使能梯度累加。梯度累加特性同样支持LazyInline编译加速。

#### 推理

自2.2版本起MindSpore主发布包不再提供配套310的推理接口使能，如需使用请切换安装MindSpore Lite发布包或下载MindSpore2.0之前的版本。MindSpore lite的安装部署与用法详见 <https://www.mindspore.cn/lite>。昇腾（Ascend）310是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式模型进行推理。原先MindSpore提供了两种在Ascend 310硬件上的推理使能用法：

1. 由MindSpore主发布包提供配套Ascend 310的版本，支持C++推理接口。
2. 由MindSpore Lite发布包提供配套Ascend的版本，支持C++/Java两种语言进行推理。

这两种方案提供的C++ API基本一致，后续不再构建和维护两套接口，而是归一使用MindSpore Lite。原有基于MindSpore主发布包构建的310推理业务，可以少量修改切换到MindSpore Lite，详见 <https://www.mindspore.cn/docs/zh-CN/r2.2/faq/inference.html>。

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

- [STABLE] 在Ascend 910系列硬件上，支持LLAMA、stable diffusion系列模型的FlashAttention大算子融合。

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

- `mindspore.dataset.GraphData`、`mindspore.dataset.Graph`、`mindspore.dataset.InMemoryGraphDataset`、`mindspore.dataset.ArgoverseDataset`不再进行功能演进并废弃。使用[MindSpore Graph Learning](https://gitee.com/mindspore/graphlearning)进行相关功能替换。对于Model仓库使用到此API的相关网络进行替换时，GCN请参考[Graph Learning GCN](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/gcn)，GAT请参考[Graph Learning GAT](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/gat)。
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

- [Stable] 全面支持动态shape，算子支持度详见[nn接口动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/master/note/dynamic_shape_nn.html)、[ops接口动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/master/note/dynamic_shape_func.html)和[算子动态shape支持情况](https://www.mindspore.cn/docs/zh-CN/master/note/dynamic_shape_primitive.html)。

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
- [STABLE] 一个发布包支持多种硬件后端：Ascend 310/310P/910、Nvidia GPU、CPU。
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

- [STABLE] 对接AllToAll单算子模式。在KernelByKernel的执行模式下，支持AllToAll算子调用。
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
- [STABLE] 支持[自动数据加速](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/dataset_autotune.html)，可以自适应调节数据处理管道的执行速度。
- [BETA] [数据处理异构加速](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/dataset_offload.html) 支持了新的数据增强操作: RandomColorAdjust、RandomSharpness和TypeCast。
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
