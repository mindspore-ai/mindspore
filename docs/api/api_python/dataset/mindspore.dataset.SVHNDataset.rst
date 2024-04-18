mindspore.dataset.SVHNDataset
=============================

.. py:class:: mindspore.dataset.SVHNDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None)

    SVHN（Street View House Numbers）数据集。

    生成的数据集有两列：`[image, label]`。`image` 列的数据类型是uint8。`label` 列的数据类型是uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 ``'train'`` 、 ``'test'`` 、 ``'extra'`` 或 ``'all'`` 。默认值： ``None`` ，读取全部样本图片。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数。默认值： ``1`` 。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `usage` 参数无效。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于SVHN数据集：**

    SVHN数据集是从谷歌街景图像中的门牌号码中获得的，由10位数字组成。

    以下是原始SVHN数据集结构。可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── svhn_dataset_dir
             ├── train_32x32.mat
             ├── test_32x32.mat
             └── extra_32x32.mat

    **引用：**

    .. code-block::

        @article{
          title={Reading Digits in Natural Images with Unsupervised Feature Learning},
          author={Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng},
          conference={NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.},
          year={2011},
          publisher={NIPS}
          url={http://ufldl.stanford.edu/housenumbers}
        }


.. include:: mindspore.dataset.api_list_vision.rst
