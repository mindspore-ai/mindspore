mindspore.dataset.WikiTextDataset
=================================

.. py:class:: mindspore.dataset.WikiTextDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析WikiText2和WikiText103数据集。

    生成的数据集有一列 `[text]` ，数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'valid' 或 'all'。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
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
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `num_samples` 参数值错误，小于0。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    **关于WikiText数据集：**

    WikiText数据集是一个包含1亿字的英语词典。
    这些样本术语来自维基百科的高级和基础文章，包括Wikitext2和Wikitext103的版本。
    对于WikiText2，分别在wiki.train.tokens中有36718个样本，在wiki.test.tokens中有4358个样本，在wiki.valid.tokens中有3760个样本。
    对于WikiText103，分别在wiki.train.tokens中有1801350个样本，wiki.test.tokens中的4358个样本，Wiki.valid.tokens中的3760个样本。

    以下是原始的WikiText数据集结构。可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── WikiText2/WikiText103
             ├── wiki.train.tokens
             ├── wiki.test.tokens
             ├── wiki.valid.tokens

    **引用：**

    .. code-block::

        @article{merity2016pointer,
          title={Pointer sentinel mixture models},
          author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
          journal={arXiv preprint arXiv:1609.07843},
          year={2016}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
