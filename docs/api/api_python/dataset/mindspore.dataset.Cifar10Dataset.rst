mindspore.dataset.Cifar10Dataset
================================

.. py:class:: mindspore.dataset.Cifar10Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    CIFAR-10数据集。

    该API目前仅支持解析二进制版本的CIFAR-10文件（CIFAR-10 binary version）。
    生成的数据集有两列：`[image, label]` 。 `image` 列的数据类型是uint8。`label` 列的数据类型是uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 ``'train'`` 、 ``'test'`` 或 ``'all'`` 。
          取值为 ``'train'`` 时将会读取50,000个训练样本，取值为 ``'test'`` 时将会读取10,000个测试样本，取值为 ``'all'`` 时将会读取全部60,000个样本。默认值： ``None`` ，读取全部样本图片。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `usage` 参数取值不为 ``'train'`` 、 ``'test'`` 或 ``'all'`` 。

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于CIFAR-10数据集：**

    CIFAR-10数据集由60000张32x32彩色图片组成，总共有10个类别，每类6000张图片。有50000个训练样本和10000个测试样本。10个类别包含飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。

    以下为原始CIFAR-10数据集的结构。您可以将数据集文件解压得到如下的文件结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── cifar-10-batches-bin
            ├── data_batch_1.bin
            ├── data_batch_2.bin
            ├── data_batch_3.bin
            ├── data_batch_4.bin
            ├── data_batch_5.bin
            ├── test_batch.bin
            ├── readme.html
            └── batches.meta.text

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
