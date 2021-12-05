mindspore.dataset.Cifar100Dataset
=================================

.. py:class:: mindspore.dataset.Cifar100Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    用于读取和解析CIFAR-100数据集的源数据文件。

    生成的数据集有三列: `[image, coarse_label, fine_label]`。 `image` 列的数据类型是uint8。 `coarse_label` 和 `fine_labels` 列的数据是uint32类型的标量。

    **参数：**

    - **dataset_dir** (str): 包含数据集文件的根目录路径。
    - **usage** (str, 可选): 指定数据集的子集，可取值为 `train` ，`test` 或 `all`。使用 `train` 参数将会读取50,000个训练样本， `test` 将会读取10,000个测试样本， `all` 将会读取全部60,000个样本（默认值为None，即全部样本图片）。
    - **num_samples** (int, 可选): 指定从数据集中读取的样本数（可以小于数据集总数，默认值为None，即全部样本图片)。
    - **num_parallel_workers** (int, 可选): 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (bool, 可选): 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **sampler** (Sampler, 可选): 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **num_shards** (int, 可选): 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选): 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选): 单节点数据缓存，能够加快数据加载和处理的速度（默认值None，即不使用缓存加速）。

    **异常：**

    - **RuntimeError:** `dataset_dir` 路径下不包含数据文件。
    - **RuntimeError:** `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError:** 同时指定了 `sampler` 和 `shuffle` 参数。
    - **RuntimeError:** 同时指定了 `sampler` 和 `num_shards` 参数。
    - **RuntimeError:** 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError:** 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError:**  `shard_id` 参数错误（小于0或者大于等于 `num_shards`）。

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

    >>> cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
    >>>
    >>> # 1)  按数据集文件的读取顺序，依次获取CIFAR-100数据集中的所有样本
    >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, shuffle=False)
    >>>
    >>> # 2)  从CIFAR100数据集中随机抽取350个样本
    >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, num_samples=350, shuffle=True)
    >>>
    >>> #  提示： 在CIFAR-100数据集生成的数据集对象中，每一次迭代得到的数据行都有"image", "fine_label" 和 "coarse_label"三个键

    **关于CIFAR-100数据集:**

    CIFAR-100数据集和CIFAR-10数据集非常相似，CIFAR-100有100个类别，每类包含600张图片，其中500张训练图片和100张测试图片。这100个类别又被分成20个超类。每个图片都有一个"fine"标签（所属子类）和一个"coarse"标签(所属超类)。

    以下为原始CIFAR-100数据集结构。您可以将数据集解压成如下的文件结构，并通过MindSpore的API进行读取。

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

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst