mindspore.dataset.CelebADataset
===============================

.. py:class:: mindspore.dataset.CelebADataset(dataset_dir, num_parallel_workers=None, shuffle=None, usage='all', sampler=None, decode=False, extensions=None, num_samples=None, num_shards=None, shard_id=None, cache=None, decrypt=None)

    读取和解析CelebA数据集的源文件构建数据集。目前仅支持解析CelebA数据集中的 `list_attr_celeba.txt` 文件作为数据集的label。

    生成的数据集有两列 `[image, attr]` 。 `image` 列的数据类型为uint8。`attr` 列的数据类型为uint32，并以one-hot编码的形式生成。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'valid'、'test' 或 'all'。默认值：'all'，全部样本图片。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **extensions** (list[str], 可选) - 指定文件的扩展名，仅读取与指定扩展名匹配的文件到数据集中。默认值：None。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **decrypt** (callable, 可选) - 图像解密函数，接受加密的图片路径并返回bytes类型的解密数据。默认值：None，不进行解密。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `usage` 参数取值不为 'train'、 'valid'、 'test'或 'all'。

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

    **关于CelebA数据集：**

    CelebFaces Attributes Dataset（CelebA）数据集是一个大规模数据集，拥有超过20万张名人图像，每个图像都有40个属性标注。此数据集包含了大量不同姿态、各种背景的图像，种类丰富、数量庞大、标注充分。数据集总体包含：

    - 10177个不同的身份
    - 202599张图像
    - 每张图像拥有5个五官位置标注，40个属性标签

    此数据集可用于各种计算机视觉任务的训练和测试，包括属性识别、检测和五官定位等。

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

    您可以将上述Anno目录下的txt文件与Img目录下的文件解压放至同一目录，并通过MindSpore的API进行读取。

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
        title         = {Deep Learning Attributes in the Wild},
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


.. include:: mindspore.dataset.api_list_vision.rst
