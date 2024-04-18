mindspore.dataset.GeneratorDataset
===================================

.. py:class:: mindspore.dataset.GeneratorDataset(source, column_names=None, column_types=None, schema=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=6)

    自定义Python数据源，通过迭代该数据源构造数据集。生成的数据集的列名和列类型取决于用户定义的Python数据源。

    参数：
        - **source** (Union[Callable, Iterable, Random Accessible]) - 一个Python的可调用对象，可以是可迭代的Python对象，或支持随机访问的Python对象。

          - 如果 `source` 是可调用对象，要求 `source` 对象可以通过 `source().next()` 的方式返回一个由NumPy数组构成的元组。
          - 如果 `source` 是可迭代对象，要求 `source` 对象通过 `iter(source).next()` 的方式返回一个由NumPy数组构成的元组。
          - 如果 `source` 是支持随机访问的对象，要求 `source` 对象通过 `source[idx]` 的方式返回一个由NumPy数组构成的元组。
        - **column_names** (Union[str, list[str]]，可选) - 指定数据集生成的列名。默认值： ``None`` ，不指定。用户可以通过此参数或 `schema` 参数指定列名。
        - **column_types** (list[mindspore.dtype]，可选) - 指定生成数据集各个数据列的数据类型。默认值： ``None`` ，不指定。
          如果未指定该参数，则自动推断类型；如果指定了该参数，将在数据输出时做类型匹配检查。
        - **schema** (Union[str, :class:`~.dataset.Schema`], 可选) - 数据格式策略，用于指定读取数据列的数据类型、数据维度等信息。
          支持传入JSON文件路径或 :class:`mindspore.dataset.Schema` 构造的对象。默认值： ``None`` 。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数/线程数（由参数 `python_multiprocessing` 决定当前为多进程模式或多线程模式）。默认值： ``1`` 。
        - **shuffle** (bool，可选) - 是否混洗数据集。只有输入的 `source` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **sampler** (Union[Sampler, Iterable]，可选) - 指定从数据集中选取样本的采样器。只有输入的 `source` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **python_multiprocessing** (bool，可选) - 启用Python多进程模式加速运算。默认值： ``True`` 。当传入 `source` 的Python对象的计算量很大时，开启此选项可能会有较好效果。
        - **max_rowsize** (int, 可选) - 指定在多进程之间复制数据时，共享内存分配的基本单位，总占用的共享内存会随着 ``num_parallel_workers`` 和 :func:`mindspore.dataset.config.set_prefetch_size` 增加而变大，仅当参数 `python_multiprocessing` 设为 ``True`` 时，此参数才会生效。默认值： ``6`` ，单位为MB。

    异常：
        - **RuntimeError** - Python对象 `source` 在执行期间引发异常。
        - **RuntimeError** - `column_names` 参数指定的列名数量与 `source` 参数输出的数据数量不匹配。
        - **ValueError** - `num_parallel_workers` 参数超过最大线程数。
        - **ValueError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **ValueError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **ValueError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **ValueError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note::
        - 如果配置 `python_multiprocessing=True` （默认值： ``True`` ） 和 `num_parallel_workers>1` （默认值：1） 表示启动了多进程方式进行数据load加速，
          此时随着数据集迭代，子进程的内存占用会逐渐增加，主要是因为自定义数据集的子进程以 Copy-On-Write 的方式获取主进程中的成员变量。
          举例：如果自定义数据集 `__init__` 函数中包含大量成员变量数据（例如：在数据集构建时加载了一个非常大的文件名列表）并且使用了多进程方式，
          那这可能会导致产生OOM的问题（总内存的预估使用量是：(子进程数量 + 1) * 父进程的内存大小）。最简单的解决方法是成员变量用非引用数据类型
          （如：Pandas、Numpy或PyArrow对象）替换Python对象（如：list / dict / int / float / string等），或者加载更少的元数据以减小成员变量，
          或者配置 `python_multiprocessing=False` 使用多线程方式。

          下面有几个类/函数可以帮助你减少成员变量的大小，你可以选择来使用：

          1. :class:`mindspore.dataset.utils.LineReader` ：在 `__init__` 函数中，使用该类初始化你的文本文件对象，然后在 `__getitem__` 函数中通过该对象按行号读取文件内容。

        - `source` 参数接收用户自定义的Python函数（PyFuncs），不要将 `mindspore.nn` 和 `mindspore.ops` 目录下或其他的网络计算算子添加
          到 `source` 中。
        - 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst


.. include:: mindspore.dataset.api_list_nlp.rst
