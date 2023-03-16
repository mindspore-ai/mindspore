mindspore.dataset.AGNewsDataset
===============================

.. py:class:: mindspore.dataset.AGNewsDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析AG News数据集的源文件构建数据集。

    生成的数据集有三列 `[index, title, description]` ，三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。默认值：None，读取全部样本。
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

    **关于AGNews数据集：**

    AG是一个大型合集，具有超过100万篇新闻文章。这些新闻文章是由ComeToMyHead在持续1年多的活动中，从2000多个新闻来源收集的。ComeToMyHead是一个学术新闻搜索引擎，自2004年7月以来一直在运营。
    数据集由学者提供，用于研究目的，如数据挖掘（聚类、分类等）、信息检索（排名、搜索等）、xml、数据压缩、数据流和任何其他非商业活动。
    AG的新闻主题类别来自于原始语料库中四个最大的类别。每个分类包含30000个训练样本和1900个测试样本。train.csv中的训练样本总数为12万，test.csv中的测试样本总数为7600。

    可以将数据集文件解压缩到以下结构中，并通过MindSpore的API读取：

    .. code-block::

        .
        └── ag_news_dataset_dir
            ├── classes.txt
            ├── train.csv
            ├── test.csv
            └── readme.txt

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
