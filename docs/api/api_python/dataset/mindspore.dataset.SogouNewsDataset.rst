mindspore.dataset.SogouNewsDataset
==================================

.. py:class:: mindspore.dataset.SogouNewsDataset(dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None, cache=None)

    读取和解析SogouNew数据集的源数据集。

    生成的数据集有三列 `[index, title, content]`，三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。默认值：None，读取全部样本。
          取值为 'train' 时将会读取45万个训练样本，取值为 'test' 时将会读取6万个测试样本，取值为 'all' 时将会读取全部51万个样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None， 读取全部样本。
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

    **关于SogouNew数据集：**

    SogouNews 数据集包括3列，分别对应类别索引（1到5）、标题和内容。
    标题和内容使用双引号(")进行转义，任何内部双引号都使用2个双引号("")进行转义。
    新行使用反斜杠进行转义，后跟“n”字符，即 "\n"。

    以下是原始SogouNew数据集结构，可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取：

    .. code-block::

        .
        └── sogou_news_dir
             ├── classes.txt
             ├── readme.txt
             ├── test.csv
             └── train.csv

    **引用：**

    .. code-block::

        @misc{zhang2015characterlevel,
            title={Character-level Convolutional Networks for Text Classification},
            author={Xiang Zhang and Junbo Zhao and Yann LeCun},
            year={2015},
            eprint={1509.01626},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
