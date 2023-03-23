mindspore.dataset.Multi30kDataset
=================================

.. py:class:: mindspore.dataset.Multi30kDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析Multi30k数据集的源文件构建数据集。

    生成的数据集有两列: `[text, translation]`。
    `text` 列的数据类型为string。
    `label` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'valid' 或 'all'。默认值：None，读取全部样本图片。
        - **language_pair** (Sequence[str, str], 可选) - 源语言与目标语言类别，可取值为 ['en', 'de'] 或 ['de', 'en']。默认值：None，表示 ['en', 'de']。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 是否混洗数据集。默认值：None，表示 Shuffle.GLOBAL 。
          如果输入False，将不进行混洗。
          如果输入True，效果与设置 Shuffle.GLOBAL 相同。
          如果输入Shuffle枚举值，效果如下表所示：

          - **Shuffle.GLOBAL**：混洗文件和文件中的数据。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **ValueError** - `usage` 参数取值不为 'train'、'test'、'valid' 或 'all'。
        - **TypeError** - 如果 `language_pair` 不为Sequence[str, str]类型。
        - **RuntimeError** - 如果 `num_samples` 小于0。
        - **RuntimeError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    **关于Multi30k数据集：**

    Multi30k是一个多语言的计算机视觉数据集，包含了约3.1万个以多种语言描述的标准图像。
    这些图像来源自Flickr数据集，每个图像都配有英语和德语的描述，以及其他多种语言。
    Multi30k常用在图像描述生成、机器翻译、视觉问答等任务的训练和测试中。

    您可以将数据集解压并构建成以下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        └── multi30k_dataset_directory
              ├── training
              │    ├── train.de
              │    └── train.en
              ├── validation
              │    ├── val.de
              │    └── val.en
              └── mmt16_task1_test
                   ├── val.de
                   └── val.en

    **引用：**

    .. code-block::

        @article{elliott-EtAl:2016:VL16,
        author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
        title     = {Multi30K: Multilingual English-German Image Descriptions},
        booktitle = {Proceedings of the 5th Workshop on Vision and Language},
        year      = {2016},
        pages     = {70--74},
        year      = 2016
        }


.. include:: mindspore.dataset.api_list_nlp.rst
