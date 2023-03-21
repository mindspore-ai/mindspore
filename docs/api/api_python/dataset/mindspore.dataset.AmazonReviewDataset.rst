mindspore.dataset.AmazonReviewDataset
=====================================

.. py:class:: mindspore.dataset.AmazonReviewDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    Amazon Review Full和Amazon Review Polarity数据集。

    生成的数据集有三列 `[label, title, content]` ，三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。
          对于Polarity数据集， 'train'将读取360万个训练样本， 'test'将读取40万个测试样本， 'all'将读取所有400万个样本。
          对于Full数据集， 'train'将读取300万个训练样本， 'test'将读取65万个测试样本， 'all'将读取所有365万个样本。默认值：None，读取所有样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
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

    **关于AmazonReview数据集：**

    Amazon Review Full数据集包括来自亚马逊的评论数据。这些数据跨越18年，包括截止至2013年3月的约3500万条评论。评论数据包括产品和用户信息、产品评级和产品评论。
    数据集主要用于文本分类，给定内容和标题，预测正确的星级评定。

    Amazon Review Polarity数据集对产品评分进行了分级，评论分数1和2视为负面评论，4和5视为正面评论。
    评分3的样本则被忽略。

    Amazon Reviews Polarity和Amazon Reviews Full datasets具有相同的目录结构。
    可以将数据集文件解压缩到以下结构，并通过MindSpore的API读取：

    .. code-block::

        .
        └── amazon_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

    **引用：**

    .. code-block::

        @article{zhang2015character,
          title={Character-level convolutional networks for text classification},
          author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          journal={Advances in neural information processing systems},
          volume={28},
          pages={649--657},
          year={2015}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
