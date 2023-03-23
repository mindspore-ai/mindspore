mindspore.dataset.QMnistDataset
===============================

.. py:class:: mindspore.dataset.QMnistDataset(dataset_dir, usage=None, compat=True, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析QMNIST数据集的源文件构建数据集。

    生成的数据集有两列: `[image, label]`。 `image` 列的数据类型为uint8。 `label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'test10k'、'test50k'、'nist' 或 'all'。默认值：None，读取所有子集。
        - **compat** (bool, 可选) - 指定每个样本的标签是类别号（compat=True）还是完整的QMNIST信息（compat=False）。默认值：True，标签为类别号。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
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

    **关于QMNIST数据集：**
    
    QMNIST 数据集是从 NIST Special Database 19 中的原始数据生成的，目的是尽可能地匹配 MNIST 预处理。
    研究人员试图生成额外的 50k 类似 MNIST 数据的图像。在QMNIST论文中，作者给出了重建过程，并使用匈牙利算法来找到原始 MNIST 样本与其重建样本之间的最佳匹配。

    以下是原始的QMNIST数据集结构。
    可以将数据集文件解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── qmnist_dataset_dir
             ├── qmnist-train-images-idx3-ubyte
             ├── qmnist-train-labels-idx2-int
             ├── qmnist-test-images-idx3-ubyte
             ├── qmnist-test-labels-idx2-int
             ├── xnist-images-idx3-ubyte
             └── xnist-labels-idx2-int

    **引用：**

    .. code-block::

        @incollection{qmnist-2019,
           title = "Cold Case: The Lost MNIST Digits",
           author = "Chhavi Yadav and L\'{e}on Bottou",\
           booktitle = {Advances in Neural Information Processing Systems 32},
           year = {2019},
           publisher = {Curran Associates, Inc.},
        }


.. include:: mindspore.dataset.api_list_vision.rst
