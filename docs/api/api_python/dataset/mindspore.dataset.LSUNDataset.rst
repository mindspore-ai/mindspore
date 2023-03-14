mindspore.dataset.LSUNDataset
=============================

.. py:class:: mindspore.dataset.LSUNDataset(dataset_dir, usage=None, classes=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析LSUN数据集的源文件构建数据集。

    生成的数据集有两列: `[image, label]` 。
    `image` 列的数据类型为uint8。
    `label` 列的数据类型为int32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可为 'train'，'test'，'valid' 或 'all'。默认值：None，将设置为 'all'。
        - **classes** (Union[str, list[str]], 可选) - 读取数据集指定的类别。默认值：None，表示读取所有类别。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部图像。
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
        - **ValueError** - `usage` 或 `classes` 参数错误（不为可选的类别）。

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

    **关于LSUN数据集：**
    
    LSUN（Large-Scale Scene Understanding）是一个大规模数据集，用于室内场景理解。LSUN最初是在2015年由斯
    坦福大学推出的，旨在为计算机视觉和机器学习领域的研究提供一个具有挑战性和多样性的数据集。该数据集的主
    要应用是室内场景分析。

    该数据集包含了十种不同的场景类别，包括卧室、客厅、餐厅、起居室、书房、厨房、浴室、走廊、儿童房和室外。
    每种类别都包含了来自不同视角的数万张图像，并且这些图像都是高质量、高分辨率的真实世界图像。

    您可以解压原始的数据集文件构建成如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── lsun_dataset_directory
            ├── test
            │    ├── ...
            ├── bedroom_train
            │    ├── 1_1.jpg
            │    ├── 1_2.jpg
            ├── bedroom_val
            │    ├── ...
            ├── classroom_train
            │    ├── ...
            ├── classroom_val
            │    ├── ...

    **引用：**

    .. code-block::

        article{yu15lsun,
            title={LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop},
            author={Yu, Fisher and Zhang, Yinda and Song, Shuran and Seff, Ari and Xiao, Jianxiong},
            journal={arXiv preprint arXiv:1506.03365},
            year={2015}
        }


.. include:: mindspore.dataset.api_list_vision.rst
