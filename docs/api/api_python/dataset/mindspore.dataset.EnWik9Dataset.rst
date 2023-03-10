mindspore.dataset.EnWik9Dataset
===============================

.. py:class:: mindspore.dataset.EnWik9Dataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=True, num_shards=None, shard_id=None, cache=None)

    读取和解析EnWik9数据集。

    生成的数据集有一列 `[text]` ，数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：True。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和样本。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    **关于EnWik9数据集：**

    EnWik9的数据是一系列UTF-8编码的XML，主要由英文文本组成。数据集包含243,426篇文章标题，其中85,560个被重定向以修复丢失的网页链接，其余是常规文章。

    数据是UTF-8格式。所有字符都在U'0000到U'10FFFF范围内，有效编码为1到4字节。字节值0xC0、0xC1和0xF5-0xFF从未出现。此外，在维基百科转储中，除了0x09（制表符）和0x0A（换行符）外，没有范围为0x00-0x1F的控制字符。
    断行符只出现在段落边界上，因此整体是有语义目的。

    可以将数据集文件解压缩到以下目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── EnWik9
             ├── enwik9

    **引用：**

    .. code-block::

        @NetworkResource{Hutter_prize,
        author    = {English Wikipedia},
        url       = "https://cs.fit.edu/~mmahoney/compression/textdata.html",
        month     = {March},
        year      = {2006}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
