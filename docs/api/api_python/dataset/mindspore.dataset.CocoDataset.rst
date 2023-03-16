mindspore.dataset.CocoDataset
==============================

.. py:class:: mindspore.dataset.CocoDataset(dataset_dir, annotation_file, task='Detection', num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None, extra_metadata=False, decrypt=None)

    读取和解析COCO数据集的源文件构建数据集。该API支持解析COCO2017数据集，支持四种类型的机器学习任务，分别是目标检测、关键点检测、物体分割和全景分割。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **annotation_file** (str) - 数据集标注JSON文件的路径。
        - **task** (str, 可选) - 指定COCO数据的任务类型。支持的任务类型包括：'Detection'、'Stuff'、'Panoptic' 和 'Keypoint'。默认值：'Detection'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None，表2中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None，表2中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **extra_metadata** (bool, 可选) - 用于指定是否额外输出一个数据列用于表示图片元信息。如果为True，则将额外输出一个名为 `[_meta-filename, dtype=string]` 的数据列。默认值：False。
        - **decrypt** (callable, 可选) - 图像解密函数，接受加密的图片路径并返回bytes类型的解密数据。默认值：None，不进行解密。

    [表1] 根据不同 `task` 参数设置，生成数据集具有不同的输出列：

    +-------------------------+----------------------------------------------+
    | `task`                  |   输出列                                     |
    +=========================+==============================================+
    | Detection               |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [bbox, dtype=float32]                      |
    |                         |                                              |
    |                         |   [category_id, dtype=uint32]                |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    +-------------------------+----------------------------------------------+
    | Stuff                   |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [segmentation, dtype=float32]              |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    +-------------------------+----------------------------------------------+
    | Keypoint                |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [keypoints, dtype=float32]                 |
    |                         |                                              |
    |                         |   [num_keypoints, dtype=uint32]              |
    +-------------------------+----------------------------------------------+
    | Panoptic                |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [bbox, dtype=float32]                      |
    |                         |                                              |
    |                         |   [category_id, dtype=uint32]                |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    |                         |                                              |
    |                         |   [area, dtype=uint32]                       |
    +-------------------------+----------------------------------------------+

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **RuntimeError** - 解析 `annotation_file` 指定的JSON文件失败。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `task` 参数取值不为 `Detection` 、 `Stuff` 、`Panoptic` 或 `Keypoint` 。
        - **ValueError** - `annotation_file` 参数对应的文件不存在。
        - **ValueError** - `dataset_dir` 参数路径不存在。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    .. note::
        - 当参数 `extra_metadata` 为True时，还需使用 `rename` 操作删除额外数据列 '_meta-filename'的前缀 '_meta-'，
          否则迭代得到的数据行中不会出现此额外数据列。
        - 暂不支持指定 `sampler` 参数为 `mindspore.dataset.PKSampler`。
        - 此数据集可以指定参数 `sampler` ，但参数 `sampler` 和参数 `shuffle` 的行为是互斥的。下表展示了几种合法的输入参数组合及预期的行为。

    .. list-table:: [表2] 配置 `sampler` 和 `shuffle` 的不同组合得到的预期排序结果
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

    **关于COCO数据集：**

    Microsoft Common Objects in Context（COCO）是一个大型数据集，该数据集专门为目标检测，语义分割和字幕生成任务而设计。它拥有330K张图像（标记数量大于200K个）、1500000个目标实例、80个目标类别、91个对象类别、每张图片均有5个字幕、带关键点标注的人有250000个。与流行的ImageNet数据集相比，COCO的类别较少，但每个类别中的图片样本非常多。

    您可以解压缩原始COCO-2017数据集文件得到如下目录结构，并通过MindSpore的API读取。

    .. code-block::

        .
        └── coco_dataset_directory
             ├── train2017
             │    ├── 000000000009.jpg
             │    ├── 000000000025.jpg
             │    ├── ...
             ├── test2017
             │    ├── 000000000001.jpg
             │    ├── 000000058136.jpg
             │    ├── ...
             ├── val2017
             │    ├── 000000000139.jpg
             │    ├── 000000057027.jpg
             │    ├── ...
             └── annotation
                  ├── captions_train2017.json
                  ├── captions_val2017.json
                  ├── instances_train2017.json
                  ├── instances_val2017.json
                  ├── person_keypoints_train2017.json
                  └── person_keypoints_val2017.json

    **引用：**

    .. code-block::

        @article{DBLP:journals/corr/LinMBHPRDZ14,
        author        = {Tsung{-}Yi Lin and Michael Maire and Serge J. Belongie and
                        Lubomir D. Bourdev and  Ross B. Girshick and James Hays and
                        Pietro Perona and Deva Ramanan and Piotr Doll{\'{a}}r and C. Lawrence Zitnick},
        title         = {Microsoft {COCO:} Common Objects in Context},
        journal       = {CoRR},
        volume        = {abs/1405.0312},
        year          = {2014},
        url           = {http://arxiv.org/abs/1405.0312},
        archivePrefix = {arXiv},
        eprint        = {1405.0312},
        timestamp     = {Mon, 13 Aug 2018 16:48:13 +0200},
        biburl        = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org}
        }


.. include:: mindspore.dataset.api_list_vision.rst
