mindspore.dataset.InMemoryGraphDataset
========================================

.. py:class:: mindspore.dataset.InMemoryGraphDataset(data_dir, save_dir="./processed", column_names="graph", num_samples=None, num_parallel_workers=1, shuffle=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=6)

    用于将图数据加载到内存中的Dataset基类。

    建议通过继承这个基类来实现自定义Dataset，并重写相应的方法，如 `process` 、 `save` 和 `load` ，可参考 `ArgoverseDataset` 源码。自定义Dataset的初始化过程如下，首先检查在给定的 `data_dir` 中是否已经有处理好的数据。如果是则调用 `load` 方法直接加载它，否则将调用 `process` 方法创建图，并调用 `save` 方法将图保存到 `save_dir`。

    可以访问所创建dataset中的图并使用，例如 `graphs = my_dataset.graphs`，也可以迭代dataset对象如 `my_dataset.create_tuple_iterator()` 来获取数据（这时需要实现 `__getitem__` 和 `__len__`）方法，具体请参考以下示例。注意：内部逻辑指定了 `__new__` 阶段会重新初始化 `__init__` ，如果自定义图实现了 `__new__` 方法，该方法将失效。

    参数：
        - **data_dir** (str) - 加载数据集的目录，这里包含原始格式的数据，并将在 `process` 方法中被加载。
        - **save_dir** (str) - 保存处理后得到的数据集的相对目录，该目录位于 `data_dir` 下面，默认值："./processed"。
        - **column_names** (Union[str, list[str]]，可选) - dataset包含的单个列名或多个列名组成的列表，默认值：'Graph'。当实现类似 `__getitem__` 等方法时，列名的数量应该等于该方法中返回数据的条数。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数/线程数（由参数 `python_multiprocessing` 决定当前为多进程模式或多线程模式），默认值：1。
        - **shuffle** (bool，可选) - 是否混洗数据集。当实现的Dataset带有可随机访问属性（ `__getitem__` ）时，才可以指定该参数。默认值：None。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数，默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号，默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **python_multiprocessing** (bool，可选) - 启用Python多进程模式加速运算，默认值：True。当传入 `source` 的Python对象的计算量很大时，开启此选项可能会有较好效果。
        - **max_rowsize** (int, 可选) - 指定在多进程之间复制数据时，共享内存分配的最大空间，默认值：6，单位为MB。仅当参数 `python_multiprocessing` 设为True时，此参数才会生效。

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.b.rst

    .. include:: mindspore.dataset.Dataset.c.rst

    .. include:: mindspore.dataset.Dataset.d.rst

    .. py:method:: load()

        从给定（处理好的）路径加载数据，也可以在自己实现的Dataset类中实现这个方法。

    .. include:: mindspore.dataset.Dataset.e.rst

    .. py:method:: process()

        与原始数据集相关的处理方法，建议在自定义的Dataset中重写此方法。

    .. include:: mindspore.dataset.Dataset.f.rst

    .. py:method:: save()

        将经过 `process` 函数处理后的数据以 numpy.npz 格式保存到磁盘中，也可以在自己实现的Dataset类中自己实现这个方法。

    .. include:: mindspore.dataset.Dataset.g.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
