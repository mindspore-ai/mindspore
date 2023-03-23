mindspore.dataset.SBUDataset
============================

.. py:class:: mindspore.dataset.SBUDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    SBU（SBU Captioned Photo）数据集。

    生成的数据集有两列：`[image, caption]`。`image` 列的数据类型为uint8。`caption` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
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

    **关于SBU数据集：**

    SBU数据集是一个带字幕的大型照片集。它包含一百万张带有视觉相关标注的图像。

    你需要使用官方的download.m手动下载图片，将 'urls{i}(24, end)'替换为 'urls{i}(24:1:end)'，并将目录保持如下。

    .. code-block::

        .
        └─ dataset_dir
           ├── SBU_captioned_photo_dataset_captions.txt
           ├── SBU_captioned_photo_dataset_urls.txt
           └── sbu_images
               ├── m_3326_3596303505_3ce4c20529.jpg
               ├── ......
               └── m_2522_4182181099_c3c23ab1cc.jpg

    **引用：**

    .. code-block::

        @inproceedings{Ordonez:2011:im2text,
          Author    = {Vicente Ordonez and Girish Kulkarni and Tamara L. Berg},
          Title     = {Im2Text: Describing Images Using 1 Million Captioned Photographs},
          Booktitle = {Neural Information Processing Systems ({NIPS})},
          Year      = {2011},
        }


.. include:: mindspore.dataset.api_list_vision.rst
