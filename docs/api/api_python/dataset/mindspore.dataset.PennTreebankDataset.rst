mindspore.dataset.PennTreebankDataset
=====================================

.. py:class:: mindspore.dataset.PennTreebankDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析PennTreebank数据集的源数据集。

    生成的数据集有一列 `[text]`。 `text` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'valid' 或 'all'。
          取值为 'train'将读取42,068个样本， 'test'将读取3,370个样本， 'test'将读取3,761个样本， 'all'将读取所有49,199个样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：`Shuffle.GLOBAL` 。
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

    **关于PennTreebank数据集：**

    Penn Treebank (PTB) 数据集，广泛用于 NLP（自然语言处理）的机器学习研究。
    PTB 不包含大写字母、数字和标点符号，其词汇表上限为10k个不重复词，与大多数现代数据集相比相对较小，可能会导致出现大量超出词汇表外的token。

    以下是原始的PennTreebank数据集结构。
    可以将数据集文件解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── PennTreebank_dataset_dir
             ├── ptb.test.txt
             ├── ptb.train.txt
             └── ptb.valid.txt

    **引用：**

    .. code-block::

        @techreport{Santorini1990,
          added-at = {2014-03-26T23:25:56.000+0100},
          author = {Santorini, Beatrice},
          biburl = {https://www.bibsonomy.org/bibtex/234cdf6ddadd89376090e7dada2fc18ec/butonic},
          file = {:Santorini - Penn Treebank tag definitions.pdf:PDF},
          institution = {Department of Computer and Information Science, University of Pennsylvania},
          interhash = {818e72efd9e4b5fae3e51e88848100a0},
          intrahash = {34cdf6ddadd89376090e7dada2fc18ec},
          keywords = {dis pos tagging treebank},
          number = {MS-CIS-90-47},
          timestamp = {2014-03-26T23:25:56.000+0100},
          title = {Part-of-speech tagging guidelines for the {P}enn {T}reebank {P}roject},
          url = {ftp://ftp.cis.upenn.edu/pub/treebank/doc/tagguide.ps.gz},
          year = 1990
        }


.. include:: mindspore.dataset.api_list_nlp.rst
