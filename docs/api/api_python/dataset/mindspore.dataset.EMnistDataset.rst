mindspore.dataset.EMnistDataset
===============================

.. py:class:: mindspore.dataset.EMnistDataset(dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    EMNIST（Extended MNIST）数据集。

    生成的数据集有两列： `[image, label]` 。 `image` 列的数据类型为uint8。 `label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **name** (str) - 按给定规则对数据集进行拆分，可以是 ``'byclass'`` 、 ``'bymerge'`` 、 ``'balanced'`` 、 ``'letters'`` 、 ``'digits'`` 或 ``'mnist'`` 。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 ``'train'`` 、 ``'test'`` 或 ``'all'`` 。
          取值为 ``'train'`` 时将会读取60,000个训练样本；取值为 ``'test'`` 时将会读取10,000个测试样本；取值为 ``'all'`` 时将会读取全部70,000个样本。默认值： ``None`` ，读取全部样本图片。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时数据集划分的片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于EMNIST数据集：**

    EMNIST数据集由一组手写字符数字组成，源自NIST特别版数据库19，并转换为与MNIST数据集直接匹配的28x28像素图像格式和数据集结构。
    有关数据集内容和转换过程的更多信息可在 https://arxiv.org/abs/1702.05373v1 上查阅。

    EMNIST按照不同的规则拆分成不同的子数据集的样本数和类数如下：

    按类拆分：814,255个样本和62个样本不平衡类。
    按合并拆分：814,255个样本和47个样本不平衡类。
    平衡拆分：131,600个样本和47个样本平衡类。
    按字母拆分：145,600个样本和26个样本平衡类。
    按数字拆分：280,000个样本和10个样本平衡类。
    MNIST: 70,000个样本符和10个样本平衡类。

    以下是原始EMNIST数据集结构。
    可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── mnist_dataset_dir
             ├── emnist-mnist-train-images-idx3-ubyte
             ├── emnist-mnist-train-labels-idx1-ubyte
             ├── emnist-mnist-test-images-idx3-ubyte
             ├── emnist-mnist-test-labels-idx1-ubyte
             ├── ...

    **引用：**

    .. code-block::

        @inproceedings{cohen2017emnist,
        title        = {EMNIST: Extending MNIST to handwritten letters},
        author       = {Cohen, Gregory and Afshar, Saeed and Tapson, Jonathan and Van Schaik, Andre},
        booktitle    = {2017 international joint conference on neural networks (IJCNN)},
        pages        = {2921--2926},
        year         = {2017},
        organization = {IEEE}
        }


.. include:: mindspore.dataset.api_list_vision.rst
