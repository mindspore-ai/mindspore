mindspore.dataset.OmniglotDataset
==================================

.. py:class:: mindspore.dataset.OmniglotDataset(dataset_dir, background=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析Omniglot数据集的源文件构建数据集。

    生成的数据集有两列: `[image, label]` 。
    `image` 列的数据类型为uint8。
    `label` 列的数据类型为int32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **background** (bool, 可选) - 是否使用 "background" 集来创建数据集，否则使用 "evaluation" 集创建数据集。默认值：None，将被设为True。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `shard_id` 参数错误，小于0或者大于等于 `num_shards` 。

    .. note:: 此数据集可以指定参数 `sampler` ，但参数 `sampler` 和参数 `shuffle` 的行为是互斥的。下表展示了几种合法的输入参数组合及预期的行为。

    .. list-table:: 配置 `sampler` 和 `shuffle` 的不同组合得到的预期排序结果
       :widths: 25 25 50
       :header-rows: 1

       * - 参数 `sampler`
         - 参数 `shuffle`
         - 预期数据顺序
       * - None
         - None
         - 随机排列
       * - None
         - True
         - 随机排列
       * - None
         - False
         - 顺序排列
       * - `sampler` 实例
         - None
         - 由 `sampler` 行为定义的顺序
       * - `sampler` 实例
         - True
         - 不允许
       * - `sampler` 实例
         - False
         - 不允许

    **关于Omniglot数据集：**

    Omniglot数据集是为开发更像人类的学习算法而设计的。它包含来自50个不同字母的1623个不同的手写字符。
    这1623个字符中的每一个都是由20个不同的人通过亚马逊的Mechanical Turk在线绘制的。每张图片都与一个
    笔画数据配对，由形如[x，y，t]的坐标、时间序列表示，时间单位为毫秒。

    您可以解压原始Omniglot数据集文件构建成如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── omniglot_dataset_directory
             ├── images_background/
             │    ├── character_class1/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── character_class2/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── ...
             ├── images_evaluation/
             │    ├── character_class1/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── character_class2/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── ...

    **引用：**

    .. code-block::

        @article{lake2015human,
            title={Human-level concept learning through probabilistic program induction},
            author={Lake, Brenden M and Salakhutdinov, Ruslan and Tenenbaum, Joshua B},
            journal={Science},
            volume={350},
            number={6266},
            pages={1332--1338},
            year={2015},
            publisher={American Association for the Advancement of Science}
        }


.. include:: mindspore.dataset.api_list_vision.rst
