mindspore.dataset.USPSDataset
=============================

.. py:class:: mindspore.dataset.USPSDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    USPS（U.S. Postal Service）数据集。

    生成的数据集有两列：`[image, label]`。`image` 列的数据类型为uint8。`label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。
          取值为 'train' 时将会读取7,291个样本，取值为 'test' 时将会读取2,007个测试样本，取值为 'all' 时将会读取全部9,298个样本。默认值：None，读取全部样本。
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
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `usage` 参数无效。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    **关于USPS数据集：**
    
    USPS是美国邮政服务公司从信封中自动扫描的数字数据集，包含总共9,298个16×16像素灰度样本。
    数据集中的图片内容已被预处理为居中和归一化，并集中了多种样式的字体。

    以下是原始的USPS数据集结构。可以将数据集文件下载并解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── usps_dataset_dir
             ├── usps
             ├── usps.t

    **引用：**

    .. code-block::

        @article{hull1994database,
          title={A database for handwritten text recognition research},
          author={Hull, Jonathan J.},
          journal={IEEE Transactions on pattern analysis and machine intelligence},
          volume={16},
          number={5},
          pages={550--554},
          year={1994},
          publisher={IEEE}
        }


.. include:: mindspore.dataset.api_list_vision.rst
