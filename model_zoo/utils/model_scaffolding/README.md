# Model Scaffolding

## 简介

Model Scaffolding 是一个用于搭建MindSpore深度学习模型程序的脚手架。主要提供了以下功能来辅助开发：

- 参数配置功能
- 华为云ModelArts运行与本地运行的统一工具
- 单机的自适应Ascend多卡训练拉起脚本

## 框架

### 统一参数配置

ModelScaffolding中使用yaml文件作为基础的参数配置文件。整个yaml文件会在`config.py`中被解析成一个python对象，并保留原有的yaml层级结构。

在具体项目中应用时，可以根据情况定义自己的yaml配置文件。

#### Yaml 文件介绍

ModelScaffolding使用`pyyaml`来解析，从而具体语法可以参[yaml1.1标准](http://yaml.org/spec/1.1/)。Yaml文件除了被解析成配置参数的python对象，还会用于构建一个命令行的参数配置，从而可以直接从命令行设定具体参数。

整个Yaml文件内容可以参考`default_config.yaml`文件，文件内被分为3个文档：

- 主配置文档，会被解析为最终的配置参数对象
- 参数描述文档，会被解析参数的解释文本，最终会体现在命令行参数的提示上
- 参数可选项文档，会被解析成参数的可选范围设定，最终用来校验命令行参数的输入

#### 命令行参数

yaml配置文件中的较为独立的配置，会被同步解析成命令行的可选参数。较为独立的配置是同时满足以下要求的配置：

- 属于顶层配置，一个大配置项的内部配置不会被解析成参数。
- 值的类型属于基础类型，例如数值，布尔值，字符串，而不是列表，字典等组合类型。

所有被解析成参数的对象，可以使用命令行选项`--help`来查看。其中会有一个`config_path`的保留选项，可以用于指定所依赖的基础配置yaml文件，后续的参数解析工作都会基于此选项指向的yaml配置文件进行解析。

解析过程使用python库`argparse`完成，从而一定成都的模糊参数匹配也是支持的，具体情况可以参考`argparse`的说明。

### Modelarts云上云下统一工具

ModelScaffolding提供了一些用于完成云上云下统一运行的基础工具包括：

- device_adapter.py 用于完成不同环境下的获取device和集群信息的工具
- moxing_adapter.py 专用于华为云ModelArts的完成云上运行封装的工具。

#### Device Adapter

主要提供用于在不同环境下获取`device_id`, `rank_size`, `rank_id`, `job_id`之类参数的工具

#### Moxing Adapter

主要提供基于ModelArts运行的封装工具，主要接口是`moxing_warpper`修饰器。将此修饰器修饰在原本的主程序上，从而完成一系列云上运行所特有的准备和收尾工作，包括：

- 根据`data_url`, `train_url`, `checkpoint_url`参数从OBS中下载程序运行的必要数据
- 根据云上环境设置`device_id`, `device_num`等参数
- 运行结束时将输出路径中的运行结果回传到`train_url`所指向的OBS路径

#### 保留配置项

为了支持云上的运行，在yaml配置文件中，我们通常会有一些保留配置，这些配置尽量保持现有的用法。

- enable_modelarts: 是否使用modelarts的模式，会控制moxing_wrapper里的一些操作是否执行
- data_url: 数据的url地址，modelarts专用，用于指向数据集的obs地址
- train_url: 训练的url地址，modelarts专用，用于指向训练的工作空间的obs地址，通常用来拷回训练结果
- checkpoint_url: checkpoint文件的url地址，modelarts专用，用于从obs上下载一个模型运行所依赖的checkpoint，通常在finetune或者evaluation的时候使用。
- data_path: 数据的本地路径，`data_url`指向的obs数据集，会被下载到这个路径供程序使用。
- output_path: 训练的本地路径，通常用于存储训练产生的结果文件，训练结束后，此路径下的内容会被传回`train_url`指向的obs地址。
- load_path: 加载checkpoint的本地路径，`checkpoint_url`指向的obs地址所存储的checkpoint会被下载到这个路径，程序中可以使用此路径来加载checkpoint
- device_target: 设备类型，云上云下通用，用于指明当前程序计划运行在那种计算平台上。
- enable_profiling: 控制是否进行profiling操作，用于进行性能调试，需要配套的主程序流程根据此选项进行相关的profiling调用，不会主动触发。

### 单机自适应Ascend多卡训练拉起脚本

具体脚本详见scripts/run_local_train.sh，脚本主要在Ascend环境中使用。可以根据传入的rank_table_file文件来自动生成对应数量，对应卡号的训练进程拉起程序。

rank_table_file的生成，可以参考[hccl_tools](../hccl_tools)来自动生成。

