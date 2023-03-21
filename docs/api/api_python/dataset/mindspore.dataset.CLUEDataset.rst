mindspore.dataset.CLUEDataset
=============================

.. py:class:: mindspore.dataset.CLUEDataset(dataset_files, task='AFQMC', usage='train', num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    CLUE（Chinese Language Understanding Evaluation）数据集。

    目前支持的CLUE分类任务包括：'AFQMC'、'TNEWS'、'IFLYTEK'、'CMNLI'、'WSC' 和 'CSL'。更多CLUE数据集的说明详见 `CLUE GitHub <https://github.com/CLUEbenchmark/CLUE>`_ 。

    参数：
        - **dataset_files** (Union[str, list[str]]) - 数据集文件路径，支持单文件路径字符串、多文件路径字符串列表或可被glob库模式匹配的字符串，文件列表将在内部进行字典排序。
        - **task** (str, 可选) - 任务类型，可取值为 'AFQMC'、'TNEWS'、'IFLYTEK'、'CMNLI'、'WSC' 或 'CSL'。默认值：'AFQMC'。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'eval'。默认值：'train'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：mindspore.dataset.Shuffle.GLOBAL。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和样本。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    根据给定的 `task` 参数 和 `usage` 配置，数据集会生成不同的输出列：

    +-------------------------+------------------------------+-----------------------------+
    | `task`                  |   `usage`                    |   输出列                    |
    +=========================+==============================+=============================+
    | AFQMC                   |   train                      |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+
    | TNEWS                   |   train                      |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [label, dtype=uint32]     |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    +-------------------------+------------------------------+-----------------------------+
    | IFLYTEK                 |   train                      |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    +-------------------------+------------------------------+-----------------------------+
    | CMNLI                   |   train                      |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+
    | WSC                     |   train                      |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         |                              |                             |
    |                         |                              |  [label, dtype=string]      |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         |                              |                             |
    |                         |                              |  [label, dtype=string]      |
    +-------------------------+------------------------------+-----------------------------+
    | CSL                     |   train                      |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+

    异常：
        - **ValueError** - `dataset_files` 参数所指向的文件无效或不存在。
        - **ValueError** - `task` 参数不为 'AFQMC'、 'TNEWS'、 'IFLYTEK'、 'CMNLI'、 'WSC' 或 'CSL'。
        - **ValueError** - `usage` 参数不为 'train'、 'test' 或 'eval'。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。

    **关于CLUE数据集：**

    CLUE，又名中文语言理解测评基准，包含许多有代表性的数据集，涵盖单句分类、句对分类和机器阅读理解等任务。

    您可以将数据集解压成如下的文件结构，并通过MindSpore的API进行读取，以 'afqmc' 数据集为例：

    .. code-block::

        .
        └── afqmc_public
             ├── train.json
             ├── test.json
             └── dev.json

    **引用：**

    .. code-block::

        @article{CLUEbenchmark,
        title   = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
        author  = {Liang Xu, Xuanwei Zhang, Lu Li, Hai Hu, Chenjie Cao, Weitang Liu, Junyi Li, Yudong Li,
                Kai Sun, Yechen Xu, Yiming Cui, Cong Yu, Qianqian Dong, Yin Tian, Dian Yu, Bo Shi, Jun Zeng,
                Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou,
                Shaoweihua Liu, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Zhenzhong Lan},
        journal = {arXiv preprint arXiv:2004.05986},
        year    = {2020},
        howpublished = {https://github.com/CLUEbenchmark/CLUE}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
