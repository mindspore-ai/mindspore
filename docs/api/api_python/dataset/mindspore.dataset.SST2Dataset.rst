mindspore.dataset.SST2Dataset
=============================

.. py:class:: mindspore.dataset.SST2Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析SST2数据集的源数据集。

    数据集中train.tsv文件和dev.tsv有两列 `[sentence, label]` 。
    数据集中test.tsv文件中有一列 `[sentence]` 。
    `sentence` 列和 `label` 列的数据类型都是string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'dev'。
          取值为 'train' 时将会读取67,349个训练样本，取值为 'test' 时将会读取1,821个测试样本，取值为 'dev' 时将会读取872个样本。默认值：None，读取train中样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：`Shuffle.GLOBAL` 。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗样本。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `shard_id` 参数值错误，小于0或者大于等于 `num_shards` 。

    **关于SST2数据集：**

    Stanford Sentiment Treebank是一个具有完全标记解析树的语料库，可以对语言中情感的合成效果进行完整的分析。
    语料库基于Pang和Lee（2005）介绍的数据集，由11855个从电影评论中提取的句子组成。它是用斯坦福解析器解析的，
    共包含215154个来自这些解析树的独特短语，每个短语都由3个人类评委进行注释。

    以下为原始SST2数据集的结构，您可以将数据集文件解压得到如下的文件结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── sst2_dataset_dir
            ├── train.tsv
            ├── test.tsv
            ├── dev.tsv
            └── original

    **引用：**

    .. code-block::

        @inproceedings{socher-etal-2013-recursive,
            title     = {Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
            author    = {Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning,
                          Christopher D. and Ng, Andrew and Potts, Christopher},
            booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
            month     = oct,
            year      = {2013},
            address   = {Seattle, Washington, USA},
            publisher = {Association for Computational Linguistics},
            url       = {https://www.aclweb.org/anthology/D13-1170},
            pages     = {1631--1642},
        }

.. include:: mindspore.dataset.api_list_vision.rst
