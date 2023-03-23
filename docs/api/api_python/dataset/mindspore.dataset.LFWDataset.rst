mindspore.dataset.LFWDataset
============================

.. py:class:: mindspore.dataset.LFWDataset(dataset_dir, task=None, usage=None, image_set=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    LFW（Labeled Faces in the Wild）数据集。

    当 `task` 为 'people' 时，生成的数据集有两列： `[image, label]` ；
    当 `task` 为 'pairs' 时，生成的数据集有三列： `[image1, image2, label]` 。
    `image` 列的数据类型为uint8。
    `image1` 列的数据类型为uint8。
    `image2` 列的数据类型为uint8。
    `label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **task** (str, 可选) - 指定读取LFW数据集的任务类型，支持 'people' 和 'pairs'。默认值：None，表示 'people'。
        - **usage** (str, 可选) - 指定数据集的子集，支持 '10fold'、'train'、'test' 和 'all'。默认值：None，将读取 'train' 和 'test' 子集。
        - **image_set** (str, 可选) - 指定读取子集的 Image Funneling 类型，支持 'original'、'funneled' 或 'deepfunneled'。默认值：None，将读取 'funneled' 子集。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部图像。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
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
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

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

    **关于LFW数据集：**
    
    LFW（Labelled Faces in the Wild）数据集是人脸识别领域最常用和广泛的开放数据集之一，
    它由美国马萨诸塞理工学院的Gary B. Huang等人于2007年发布。该数据集包含13,233个人的
    近50,000张图像，这些图像来自互联网上不同来源的人物照片，并包含了不同的姿势、光照、
    角度等不同环境因素。该数据集中大部分图像都是正面正视的，而且包含多种年龄、性别和人种。

    你可以将原始的LFW数据集文件解压成以下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── lfw_dataset_directory
            ├── lfw
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw-deepfunneled
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw_funneled
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw-names.txt
            ├── pairs.txt
            ├── pairsDevTest.txt
            ├── pairsDevTrain.txt
            ├── people.txt
            ├── peopleDevTest.txt
            ├── peopleDevTrain.txt

    **引用：**

    .. code-block::

        @TechReport{LFWTech,
            title={LFW: A Database for Studying Recognition in Unconstrained Environments},
            author={Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
            institution ={University of Massachusetts, Amherst},
            year={2007}
            number={07-49},
            month={October},
            howpublished = {http://vis-www.cs.umass.edu/lfw}
        }


.. include:: mindspore.dataset.api_list_vision.rst
