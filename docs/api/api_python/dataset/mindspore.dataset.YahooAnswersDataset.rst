mindspore.dataset.YahooAnswersDataset
=====================================

.. py:class:: mindspore.dataset.YahooAnswersDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析YahooAnswers数据集的源数据集。

    生成的数据集有四列 `[class, title, content, answer]` ，数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。
          取值为 'train' 时将会读取1,400,000个训练样本，取值为 'test' 时将会读取60,000个测试样本，取值为 'all' 时将会读取全部1,460,000个样本。默认值：None，读取全部样本。
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
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    **关于YahooAnswers数据集：**

    YahooAnswers数据集包含10个类的63万个文本样本。
    train.csv中有56万个样本，test.csv中有7万个样本。
    这10个不同的类代表社会与文化、科学与数学、健康、教育与参考、计算机与互联网、体育、商业与金融、娱乐与音乐、家庭与关系、政治与政府。

    以下是原始的YahooAnswers数据集结构，可以将数据集文件解压缩到此目录结构中，并由Mindspore的API读取。

    .. code-block::

        .
        └── yahoo_answers_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    **引用：**

    .. code-block::

        @article{YahooAnswers,
        title   = {Yahoo! Answers Topic Classification Dataset},
        author  = {Xiang Zhang},
        year    = {2015},
        howpublished = {}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
