mindspore.dataset.GeneratorDataset
===================================

.. py:class:: mindspore.dataset.GeneratorDataset(source, column_names=None, column_types=None, schema=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=None, batch_sampler=None, collate_fn=None)

    自定义Python数据源，通过迭代该数据源构造数据集。生成的数据集的列名和列类型取决于用户定义的Python数据源。

    参数：
        - **source** (Union[Random Accessible, Iterable]) - 自定义数据集，表示从此数据对象加载数据。MindSpore支持两种类型的数据集。

          - 可随机访问(map-style)数据集：它是一种实现了 `__getitem__()` 和 `__len__()` 方法的数据集对象，记录从索引/键到数据样本的映射。
            例如，使用 `source[idx]` 访问数据集时，可以从磁盘上的文件夹中读取第idx个样本，详情请参阅 `可随机访问数据集样例 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html#可随机访问数据集>`_ 。
          - 可迭代(iterable-style)数据集：它是一种实现了 `__iter__()` 和 `__next__()` 方法的数据集对象，表示数据样本的可迭代性。这种类型的数据集适用于随机读取成本较高甚至不可能的情况，以及适用于批量大小取决于获取数据的情况。
            例如，使用 `iter(source)` 访问数据集时，可以返回从数据库、远程服务器读取的数据流，详情请参阅 `可迭代数据集样例 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html#可迭代数据集>`_ 。
        - **column_names** (Union[str, list[str]]，可选) - 指定数据集生成的列名。默认值： ``None`` ，不指定。用户可以通过此参数或 `schema` 参数指定列名。
        - **column_types** (list[mindspore.dtype]，可选) - 指定生成数据集各个数据列的数据类型。默认值： ``None`` ，不指定。
          如果未指定该参数，则自动推断类型；如果指定了该参数，将在数据输出时做类型匹配检查（后续版本将废弃此参数）。
        - **schema** (Union[str, :class:`~.dataset.Schema`], 可选) - 数据格式策略，用于指定读取数据列的数据类型、数据维度等信息。
          支持传入JSON文件路径或 :class:`mindspore.dataset.Schema` 构造的对象（后续版本将废弃此参数）。默认值： ``None`` 。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数/线程数（由参数 `python_multiprocessing` 决定当前为多进程模式或多线程模式）。默认值： ``1`` 。
        - **shuffle** (bool，可选) - 是否混洗数据集。只有输入的 `source` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **sampler** (Union[Sampler, Iterable]，可选) - 指定从数据集中选取样本的采样器。只有输入的 `source` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时数据集划分的片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **python_multiprocessing** (bool，可选) - 启用Python多进程模式加速运算。默认值： ``True`` 。当传入 `source` 的Python对象的计算量很大时，开启此选项可能会有较好效果。
        - **max_rowsize** (int, 可选) - 指定在多进程之间复制数据时，共享内存分配的基本单位，单位为MB，总占用的共享内存会随着 ``num_parallel_workers`` 和 :func:`mindspore.dataset.config.set_prefetch_size` 增加而变大。如果设置为 ``-1`` ，共享内存将随数据大小动态分配。仅当参数 `python_multiprocessing` 设为 ``True`` 时，此参数才会生效。默认值： ``None`` ，动态分配共享内存（后续版本将废弃此参数）。
        - **batch_sampler** (Iterable，可选) - 与 `sampler` 类似，但每次返回1批索引，对应的数据将被合并为1个Batch。不可与 `num_samples` 、`shuffle` 、`num_shards` 、`shard_id` 和 `sampler` 等参数同时使用。默认值： ``None`` ，不使用批采样器。
        - **collate_fn** (Callable[List[numpy.ndarray]]，可选) - 定义如何将数据列表合并为1个Batch。仅在使用了 `batch_sampler` 时有效。默认值：``None`` ，不指定合并函数。

    .. warning::
        在多进程 `spawn` 模式下， `GeneratorDataset` 会隐式使用 `dill` 模块对 `source` 进行序列化/反序列化，而该模块存在已知安全隐患。
        攻击者可构造恶意 pickle 数据，在反序列化过程中执行任意代码。切勿加载可能来自不可信来源或已被篡改的数据。

    .. note::
        - 参数 `column_types` 、 `schema` 和 `max_rowsize` 将在后续版本废弃。
        - 如果配置 `python_multiprocessing=True` （默认值： ``True`` ） 和 `num_parallel_workers>1` （默认值：1） 表示启动了多进程方式进行数据load加速，
          此时随着数据集迭代，子进程的内存占用会逐渐增加，主要是因为自定义数据集的子进程以 Copy-On-Write 的方式获取主进程中的成员变量。
          举例：如果自定义数据集 `__init__` 函数中包含大量成员变量数据（例如：在数据集构建时加载了一个非常大的文件名列表）并且使用了多进程方式，
          那这可能会导致产生OOM的问题（总内存的预估使用量是：(子进程数量 + 1) * 父进程的内存大小）。最简单的解决方法是成员变量用非引用数据类型
          （如：Pandas、Numpy或PyArrow对象）替换Python对象（如：list / dict / int / float / string等），或者加载更少的元数据以减小成员变量，
          或者配置 `python_multiprocessing=False` 使用多线程方式。

          你可以使用以下类/函数来减少成员变量的大小：

          - :class:`mindspore.dataset.utils.LineReader` ：在 `__init__` 函数中，使用该类初始化你的文本文件对象，然后在 `__getitem__` 函数中通过该对象按行号读取文件内容。

        - `source` 参数接收用户自定义的Python函数（PyFuncs），通过ds.config.set_multiprocessing_start_method("spawn")方式设置多进程的启动方式为 \
          `spawn` 模式，且 `python_multiprocessing=True` 和 `num_parallel_workers>1` 时，支持将 `mindspore.nn` 和 `mindspore.ops`\
          目录下或其他的网络计算算子添加到 `source` 中，否则不支持添加到 `source` 中。
        - 当 `source` 自定义数据集对象在数据集加载及处理时，调用了DVPP算子，那么支持的场景如下：

          +----------+----------------------------+----------------------------+----------------------------+
          |          |                            |                          多进程                         |
          |          |            多线程          +----------------------------+----------------------------+
          |          |                            |            spawn           |            fork            |
          +==========+============================+============================+============================+
          |          |数据处理：支持              |数据处理：支持              |数据处理：支持              |
          |独立进程  |                            |                            |                            |
          |          |数据处理 + 网络训练：不支持 |数据处理 + 网络训练：支持   |数据处理 + 网络训练：不支持 |
          +----------+----------------------------+----------------------------+----------------------------+
          |          |数据处理：支持              |数据处理：支持              |数据处理：支持              |
          |非独立进程|                            |                            |                            |
          |          |数据处理 + 网络训练：支持   |数据处理 + 网络训练：支持   |数据处理 + 网络训练：不支持 |
          +----------+----------------------------+----------------------------+----------------------------+

        - 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    异常：
        - **RuntimeError** - Python对象 `source` 在执行期间引发异常。
        - **RuntimeError** - `column_names` 参数指定的列名数量与 `source` 参数输出的数据数量不匹配。
        - **ValueError** - `num_parallel_workers` 参数超过最大线程数。
        - **ValueError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **ValueError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **ValueError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **ValueError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - 如果 `batch_sampler` 与 `num_samples` 、`shuffle` 、`num_shards` 、`shard_id` 和 `sampler` 同时指定。
        - **ValueError** - 如果在指定 `collate_fn` 时没有指定 `batch_sampler` 。
        - **TypeError** - 如果 `batch_sampler` 不为可迭代类型。
        - **TypeError** - 如果 `collate_fn` 不为可调用函数。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

.. include:: mindspore.dataset.api_list_nlp.rst
