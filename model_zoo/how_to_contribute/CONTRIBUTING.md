# 如何贡献MindSpore ModelZoo

<!-- TOC -->

- [如何贡献MindSpore ModelZoo](#如何贡献mindspore-modelzoo)
    - [准备工作](#准备工作)
        - [了解贡献协议与流程](#了解贡献协议与流程)
        - [确定自己贡献的目标](#确定自己贡献的目标)
    - [代码提交](#代码提交)
        - [CodeStyle](#codestyle)
        - [目录结构](#目录结构)
        - [ReadMe 说明](#readme-说明)
        - [关于第三方引用](#关于第三方引用)
            - [引用额外的python库](#引用额外的python库)
            - [引用第三方开源代码](#引用第三方开源代码)
            - [引用其他系统库](#引用其他系统库)
        - [提交自检列表](#提交自检列表)
    - [维护与交流](#维护与交流)

<!-- TOC -->

本指导用于明确ModelZoo贡献规范，从而确保众多的开发者能够以一种相对统一的风格和流程参与到ModelZoo的建设中。

## 准备工作

### 了解贡献协议与流程

你应该优先参考MindSpore的[CONTRIBUTE.md](../CONTRIBUTING.md)说明来理解MindSpore的开源协议和运作方式，并确保自己已完成CLA的签署。

<!--
### 确定自己贡献的目标

如果希望进行贡献，我们推荐你先从一些较为容易的issue开始尝试。你可以从以下列表中寻找一些简单的例如bugfix的工作。

- [wanted bugfix](https://gitee.com/mindspore/mindspore/issues?assignee_id=&author_id=&branch=&issue_search=&label_ids=58021213&label_text=kind/bug&milestone_id=&program_id=&scope=&sort=newest&state=open)

如果你可以完成独立的网络贡献，你可以从以下列表中找到我们待实现的网络列表。

- [wanted implement](https://gitee.com/mindspore/mindspore/issues?assignee_id=&author_id=&branch=&issue_search=&label_ids=58022151&label_text=device%2Fascend&milestone_id=&program_id=&scope=&sort=newest&state=open)

> **Notice** 记得在选定issue之后进行一条回复，从而让别人知道你已经着手于此issue的工作。当你完成某项工作后，也记得回到issue更新你的成果。如果过程中有什么问题，也可以随时在issue中更新你的进展。
-->

## 代码提交

### CodeStyle

参考[CONTRIBUTE.md](../CONTRIBUTING.md)中关于CodeStyle的说明，你应该确保自己的代码与MindSpore的现有代码风格保持一致。

### 目录结构

为了保证ModelZoo中的实现能够提供一种相对统一的使用方法，我们提供了一种基础的**目录结构模板**，你应该基于此结构来组织自己的工程。

```shell
model_zoo
├── official                                             # 官方支持模型
│   └── XXX                   # 模型名
│       ├── README.md                # 模型说明文档
│       ├── eval.py                 # 精度验证脚本
│       ├── export.py                # 推理模型导出脚本
│       ├── scripts                 # 脚本文件
│       │   ├── run_distributed_train.sh      # 分布式训练脚本
│       │   ├── run_eval.sh             # 验证脚本
│       │   └── run_standalone_train.sh       # 单机训练脚本
│       ├── src                   # 模型定义源码目录
│       │   ├── XXXNet.py              # 模型结构定义
│       │   ├── callback.py             # 回调函数定义
│       │   ├── config.py              # 模型配置参数文件
│       │   └── dataset.py             # 数据集处理定义
│       ├── ascend_infer               # （可选）用于在Ascend推理设备上进行离线推理的脚本
│       ├── third_party               # （可选）第三方代码
│       │   └── XXXrepo               # （可选）完整克隆自第三方仓库的代码
│       └── train.py                # 训练脚本
├── research                    # 非官方研究脚本
├── community                    # 合作方脚本链接
└── utils                      # 模型通用工具
```

你可以参照以下原则，根据自己的需要在模板基础上做一些适配自己实现的修改

1. 模型根目录下只放置带有`main方法`的可执行脚本，模型的定义文件统一放在`src`目录下，该目录下可以根据自己模型的复杂程度自行组织层次结构。
2. 配置参数应当与网络定义分离，将所有可配置的参数抽离到`src/config.py`文件中统一定义。
3. 上传内容应当只包含脚本、代码和文档，**不要上传**任何数据集或checkpoint之类的数据文件。
4. third_party用于存放需要引用的第三方代码，但是不要直接将代码拷贝到目录下上传，而应该使用git链接的形式，在使用时下载。
5. 每个模型的代码应当自成闭包，可以独立的迁移使用，不应当依赖模型目录以外的其他代码。utils内只是通用工具，并非通用函数库。
6. 上传内容中**不要包含**任何你的个人信息，例如你的主机IP，个人密码，本地目录等。

### ReadMe 说明

每个AI模型都需要一个对应的`README.md`作为说明文档，对当前的模型实现进行介绍，从而向其他用户传递以下信息：

1. 这是个什么模型？来源和参考是什么？
2. 当前的实现包含哪些内容？
3. 如何使用现有的实现？
4. 这个模型表现如何？

对此，我们提供了一个基础的[README模版](./README_template.md)，你应该参考此模版来完善自己的说明文档, 也可以参考其他现有模型的readme。

### 关于第三方引用

#### 引用额外的python库

确保将自己所需要的额外python库和对应版本（如果有明确要求）注明在`requirements.txt`文件。你应该优先选择和MindSpore框架兼容的第三方库。

#### 引用第三方开源代码

你应该保证所提交的代码是自己原创开发所完成的。

当你需要借助一些开源社区的力量，应当优先引用一些成熟可信的开源项目，同时确认自己所选择的开源项目所使用的开源协议是否符合要求。

当你使用开源代码时，正确的使用方式是通过git地址获取对应代码，并在使用中将对应代码归档在独立的`third_party`目录中，保持与自己的代码隔离。**切勿粗暴的拷贝对应代码片段到自己的提交中。**

#### 引用其他系统库

你应该减少对一些独特系统库的依赖，因为这通常意味着你的提交在不同系统中难以复用。

当你确实需要使用一些独特的系统依赖来完成任务时，你需要在说明中指出对应的获取和安装方法。

### 提交自检列表

你所提交的代码应该经过充分的Review, 可以参考以下checklist进行自查

- [ ] 代码风格符合规范
- [ ] 代码在必要的位置添加了注释
- [ ] 文档已同步修改
- [ ] 同步添加了必要的测试用例
- [ ] 进行了代码自检
- [ ] 工程组织结构符合[目录结构](#目录结构)中的要求。

## 维护与交流

我们十分感谢您对MindSpore社区的贡献，同时十分希望您能够在完成一次提交之后持续关注您所提交的代码。 您可以在所提交模型的README中标注您的署名与常用邮箱等联系方式，并持续关注您的gitee、github信息。

其他的开发者也许会用到您所提交的模型，使用期间可能会产生一些疑问，此时就可以通过issue、站内信息、邮件等方式与您进行详细的交流.
