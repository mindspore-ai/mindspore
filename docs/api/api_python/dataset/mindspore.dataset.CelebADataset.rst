mindspore.dataset.CelebADataset
===============================

.. py:class:: mindspore.dataset.CelebADataset(dataset_dir, num_parallel_workers=None, shuffle=None, usage='all', sampler=None, decode=False, extensions=None, num_samples=None, num_shards=None, shard_id=None, cache=None)

    用于读取和解析CelebA数据集的源数据文件。目前仅支持读取解析标注文件 `list_attr_celeba.txt` 作为数据集的标注。

    生成的数据集有两列：`[image, attr]`。列: `image` 的数据类型为uint8。列: `attr` 的数据类型为uint32，并以one-hot编码的形式生成。

    **参数：**

    - **dataset_dir** (str) - 包含数据集文件的根目录路径。
    - **num_parallel_workers** (int，可选) - 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (bool，可选) - 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **usage** (str，可选) - 指定数据集的子集，可取值为'train'，'valid'，'test'或'all'。（默认值为'all'，即全部样本图片）。
    - **sampler** (Sampler，可选) - 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **decode** (bool，可选) - 是否对读取的图像进行解码操作（默认为False）。
    - **extensions** (list[str]，可选) - 指定文件扩展后缀，仅读取这些后缀的文件到数据集中（默认为None）。
    - **num_samples** (int，可选) - 指定从数据集中读取的样本数（可以小于数据集总数，默认值为None，即全部样本图片)。
    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数（默认值None）。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。

    **异常：**

    - **RuntimeError** - 参数 `dataset_dir` 不包含任何数据文件。
    - **RuntimeError** - 参数 `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 。
    - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError** - `shard_id` 参数错误（小于0或者大于等于 `num_shards` ）。

    .. note:: 此数据集可以指定 `sampler` 参数，但 `sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

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
         - None
         - 由 `sampler` 行为定义的顺序
       * - 参数 `sampler`
         - True
         - 不允许
       * - 参数 `sampler`
         - False
         - 不允许

    **样例：**

    >>> celeba_dataset_dir = "/path/to/celeba_dataset_directory"
    >>>
    >>> # 从CelebA数据集中随机读取5个样本图片
    >>> dataset = ds.CelebADataset(dataset_dir=celeba_dataset_dir, usage='train', num_samples=5)
    >>>
    >>> # 注：在生成的数据集对象中，每一次迭代得到的数据行都有"image"和"attr" 两个键

    **关于CelebA数据集：**

    CelebFaces Attributes Dataset（CelebA）数据集是一个大规模的人脸属性数据集，拥有超过20万名人图像，每个图像都有40个属性标注。此数据集包含了大量不同姿态、各种背景的人脸图像，种类丰富、数量庞大、标注充分。数据集总体包含：

    - 10177个不同的身份
    - 202599张人脸图像
    - 每张图像拥有5个五官位置标注，40个属性标签。

    此数据集可用于各种计算机视觉任务的训练和测试，包括人脸识别、人脸检测、五官定位、人脸编辑和合成等。

    原始CelebA数据集结构：

    .. code-block::

        .
        └── CelebA
             ├── README.md
             ├── Img
             │    ├── img_celeba.7z
             │    ├── img_align_celeba_png.7z
             │    └── img_align_celeba.zip
             ├── Eval
             │    └── list_eval_partition.txt
             └── Anno
                  ├── list_landmarks_celeba.txt
                  ├── list_landmarks_align_celeba.txt
                  ├── list_bbox_celeba.txt
                  ├── list_attr_celeba.txt
                  └── identity_CelebA.txt

    您可以将数据集解压成如下的文件结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── celeba_dataset_directory
            ├── list_attr_celeba.txt
            ├── 000001.jpg
            ├── 000002.jpg
            ├── 000003.jpg
            ├── ...

    **引用：**

    .. code-block::

        @article{DBLP:journals/corr/LiuLWT14,
        author        = {Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang},
        title         = {Deep Learning Face Attributes in the Wild},
        journal       = {CoRR},
        volume        = {abs/1411.7766},
        year          = {2014},
        url           = {http://arxiv.org/abs/1411.7766},
        archivePrefix = {arXiv},
        eprint        = {1411.7766},
        timestamp     = {Tue, 10 Dec 2019 15:37:26 +0100},
        biburl        = {https://dblp.org/rec/journals/corr/LiuLWT14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org},
        howpublished  = {http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html}
        }

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst