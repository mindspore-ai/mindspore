mindspore.dataset.CocoDataset
==============================

.. py:class:: mindspore.dataset.CocoDataset(dataset_dir, annotation_file, task='Detection', num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None, extra_metadata=False)

    用于读取和解析COCO数据集的源数据文件。该API支持解析COCO2017数据集，支持四种类型的机器学习任务，分别是目标检测、关键点检测、物体分割和全景分割。

    根据不同 `task` 参数设置，生成数据集具有不同的输出列：

    - `task` = `Detection`, 输出列: `[image, dtype=uint8]`, `[bbox, dtype=float32]`, `[category_id, dtype=uint32]`, `[iscrowd, dtype=uint32]`。
    - `task` = `Stuff`, 输出列: `[image, dtype=uint8]`, `[segmentation,dtype=float32]`, `[iscrowd,dtype=uint32]`。
    - `task` = `Keypoint`, 输出列: `[image, dtype=uint8]`, `[keypoints, dtype=float32]`, `[num_keypoints, dtype=uint32]`。
    - `task` = `Panoptic`, 输出列: `[image, dtype=uint8]`, `[bbox, dtype=float32]`, `[category_id, dtype=uint32]`, `[iscrowd, dtype=uint32]`, `[area, dtype=uint32]`。

    **参数：**

    - **dataset_dir** (str) - 包含数据集文件的根目录路径。
    - **annotation_file** (str) - 数据集标注JSON文件的路径。
    - **task** (str，可选) - 指定COCO数据的任务类型。支持的任务类型包括：`Detection` 、 `Stuff` 、 `Panoptic` 和 `Keypoint` （默认为 `Detection` ）。
    - **num_samples** (int，可选) - 指定从数据集中读取的样本数（可以小于数据集总数，默认值为None，即全部样本图片)。
    - **num_parallel_workers** (int，可选) - 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (bool，可选) - 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **decode** (bool，可选) - 是否对读取的图像进行解码操作（默认为False）。
    - **sampler** (Sampler，可选) - 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数（默认值None）。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。
    - **extra_metadata** (bool，可选) - 用于指定是否额外输出一列数据用于表示图像元信息。如果为True，则将额外输出一列数据，名为 `[_meta-filename, dtype=string]` （默认值为False）。

    **异常：**

    - **RuntimeError** - 参数 `dataset_dir` 不包含任何数据文件。
    - **RuntimeError** - 参数 `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 。
    - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **RuntimeError** - 解析JSON文件失败。
    - **ValueError** - 指定的任务不为 `Detection` ， `Stuff` ， `Panoptic` 或 `Keypoint`。
    - **ValueError** - 参数 `annotation_file` 对应的文件不存在。
    - **ValueError** - 参数 `dataset_dir` 路径不存在。
    - **ValueError** - 参数 `shard_id` 错误（小于0或者大于等于 `num_shards` ）。

    .. note::
        - 当指定 `extra_metadata` 为True时，除非显式使用 `rename` 算子以删除元信息列明的前缀('_meta-')，否则迭代的数据行中不会出现'[_meta-filename, dtype=string]'列。
        - CocoDataset的 `sampler` 参数不支持指定PKSampler。
        - 此数据集可以指定 `sampler` 参数，但 `sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

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
       * - 参数 `sampler`
         - No ne
         - 由 `sampler` 行为定义的顺序
       * - 参数 `sampler`
         - True
         - 不允许
       * - 参数 `sampler`
         - False
         - 不允许

    **样例：**

    >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
    >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
    >>>
    >>> # 1）读取COCO数据集中 `Detection` 任务中的数据。
    >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
    ...                          annotation_file=coco_annotation_file,
    ...                          task='Detection')
    >>>
    >>> # 2）读取COCO数据集中 `Stuff` 任务中的数据。
    >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
    ...                          annotation_file=coco_annotation_file,
    ...                          task='Stuff')
    >>>
    >>> # 3）读取COCO数据集中 `Panoptic` 任务中的数据。
    >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
    ...                          annotation_file=coco_annotation_file,
    ...                          task='Panoptic')
    >>>
    >>> # 4）读取COCO数据集中 `Keypoint` 任务中的数据。
    >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
    ...                          annotation_file=coco_annotation_file,
    ...                          task='Keypoint')
    >>>
    >>> # 在生成的COCO数据集对象中，每一次迭代得到的数据行都有"image"和"annotation"两个键。

    **关于COCO数据集：**

    Microsoft Common Objects in Context（COCO）是一个大型数据集，该数据集专门为目标检测，语义分割和字幕生成任务而设计。它拥有330K张图像（标记数量大于200K个）、1500000个目标实例、80个目标类别、91个对象类别、每张图片均有5个字幕、带关键点标注的人有250000个。与流行的ImageNet数据集相比，COCO的类别较少，但每个类别中的图片样本非常多。

    您可以解压缩原始COCO-2017数据集文件如下目录结构，并通过MindSpore的API读取。

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

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst