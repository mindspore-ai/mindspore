mindspore.dataset.SemeionDataset
================================

.. py:class:: mindspore.dataset.SemeionDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    Semeion数据集。

    生成的数据集有两列：`[image, label]`。`image` 列的数据类型为uint8。`label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
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

    **关于SEMEION数据集：**

    该数据集由意大利布雷西亚Tactile Srl创建（http://www.tattil.it），并于1994年捐赠给意大利罗马Semeion通信科学研究中心（http://www.semeion.it），用于机器学习研究。
    此数据集由1593条样本记录（行）和256个属性（列）组成。每条记录代表一个手写数字，最初扫描的分辨率为256灰度。
    数据集拉伸了每个原始扫描图像的每个像素，然后在0和1之间缩放（将值低于灰度值127的每个像素（包括127）设置为0，并将灰度值超过127的每个像素设置为1）。
    最后，每个二进制图像再次缩放为一个16x16的方形图像。

    .. code-block::

        .
        └── semeion_dataset_dir
            └──semeion.data
            └──semeion.names

    **引用：**

    .. code-block::

        @article{
          title={The Theory of Independent Judges, in Substance Use & Misuse 33(2)1998, pp 439-461},
          author={M Buscema, MetaNet},
        }


.. include:: mindspore.dataset.api_list_vision.rst
