mindspore.dataset.Cifar100Dataset
=================================

.. py:class:: mindspore.dataset.Cifar100Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析CIFAR-100数据集的源文件构建数据集。

    生成的数据集有三列: `[image, coarse_label, fine_label]` 。 `image` 列的数据类型为uint8。 `coarse_label` 和 `fine_labels` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。
          取值为 'train'时将会读取50,000个训练样本，取值为 'test'时将会读取10,000个测试样本，取值为 'all'时将会读取全部60,000个样本。默认值：None，读取全部样本图片。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
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
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `usage` 参数取值不为 'train'、 'test'或 'all'。

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

    **关于CIFAR-100数据集：**

    CIFAR-100数据集和CIFAR-10数据集非常相似，CIFAR-100有100个类别，每类包含600张图片。其中500张训练图片和100张测试图片。这100个类别又被分成20个超类。每个图片都有一个"fine"标签（所属子类）和一个"coarse"标签(所属超类)。

    以下为原始CIFAR-100数据集的结构。您可以将数据集文件解压得到如下的文件结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── cifar-100-binary
            ├── train.bin
            ├── test.bin
            ├── fine_label_names.txt
            └── coarse_label_names.txt

    **引用：**

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }


.. include:: mindspore.dataset.api_list_vision.rst
