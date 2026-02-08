mindspore.dataset
=================

.. toctree::
   :maxdepth: 1
   :hidden:

   mindspore.dataset.loading
   mindspore.dataset.transforms
   mindspore.mindrecord
   mindspore.dataset.dataloader

MindSpore Dataset是MindSpore框架中专门设计的高性能数据引擎模块，致力于为深度学习任务提供高效、灵活且易用的数据加载与预处理解决方案。它支持多种数据格式（如MindRecord、TFRecord等），并内置了丰富的公开数据集接口，帮助用户快速构建数据流水线。
通过 MindSpore Dataset，用户可以轻松实现数据读取、转换、增强等操作，满足图像、文本、音频等多种数据类型的处理需求。

并且，MindSpore Dataset提供了强大的数据变换功能，支持多种数据增强操作（如裁剪、旋转、归一化等），能够有效提升模型的泛化能力。结合MindRecord高效数据存储格式，用户可进一步优化数据读取性能，显著加速大规模数据训练任务。
MindSpore Dataset的设计兼顾了灵活性与性能，支持单机与分布式训练场景，能够无缝集成到 MindSpore 的模型开发与训练流程中，为用户提供从数据预处理到模型训练的全流程高效支持。

- 数据集读取与加载（ `mindspore.dataset <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.loading.html>`_ ），
  该模块提供了多种数据加载方式，帮助用户加载数据集到MindSpore中。

- 数据增强（ `mindspore.dataset.transforms <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.transforms.html>`_ ），
  该模块提供了图像、文本、音频领域的常用数据变换，并支持自定义的数据变换，帮助用户在线完成数据增强。

- MindRecord数据格式（ `mindspore.mindrecord <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mindrecord.html>`_ ），
  该模块提供了一种高效数据格式，帮助用户方便地将数据源转为标准格式的数据文件，并在训练时高速读取。

除此之外，MindSpore还提供了一套兼容PyTorch DataLoader的接口，用户可以一键迁移PyTorch的代码到MindSpore中。

- 数据加载器（ `mindspore.dataset.dataloader <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.dataloader.html>`_ ），
  该模块提供了一系列数据加载接口，帮助用户高效地加载和处理数据。
