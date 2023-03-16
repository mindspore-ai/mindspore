mindspore.dataset.UDPOSDataset
==============================

.. py:class:: mindspore.dataset.UDPOSDataset(dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None, cache=None)

    读取和解析UDPOS数据集的源数据集。

    生成的数据集有三列 `[word, universal, stanford]`，三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'valid' 或 'all'。
          取值为 'train' 时将会读取12,543个样本，取值为 'test' 时将会读取2,077个测试样本，取值为 'valid' 时将会读取2,002个样本，取值为 'all' 时将会读取全部16,622个样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值： `Shuffle.GLOBAL` 。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和样本。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    **关于UDPOS数据集：**

    UDPOS是一个解析的文本语料库数据集，用于阐明句法或者语义句子结构。
    该语料库包含254,830个单词和16,622个句子，取自各种网络媒体，包括博客、新闻组、电子邮件和评论。

    **引用：**

    .. code-block::

        @inproceedings{silveira14gold,
          year = {2014},
          author = {Natalia Silveira and Timothy Dozat and Marie-Catherine de Marneffe and Samuel Bowman
            and Miriam Connor and John Bauer and Christopher D. Manning},
          title = {A Gold Standard Dependency Corpus for {E}nglish},
          booktitle = {Proceedings of the Ninth International Conference on Language
            Resources and Evaluation (LREC-2014)}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
