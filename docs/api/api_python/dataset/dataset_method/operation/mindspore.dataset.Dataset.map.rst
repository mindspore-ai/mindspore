mindspore.dataset.Dataset.map
===============================

.. py:method:: mindspore.dataset.Dataset.map(operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs)

    给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。

    每个数据增强操作将数据集对象中的一个或多个数据列作为输入，将数据增强的结果输出为一个或多个数据列。
    第一个数据增强操作将 `input_columns` 中指定的列作为输入。
    如果数据增强列表中存在多个数据增强操作，则上一个数据增强的输出列将作为下一个数据增强的输入列。

    最后一个数据增强的输出列的列名由 `output_columns` 指定，如果没有指定 `output_columns` ，输出列名与 `input_columns` 一致。

    - 如果使用的是 `mindspore` `dataset` 提供的数据增强（
      `vision类 <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.transforms.html#视觉>`_ 、
      `nlp类 <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.transforms.html#文本>`_ 、
      `audio类 <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.transforms.html#音频>`_ ），请使用如下参数：

      .. image:: map_parameter_cn.png

    - 如果使用的是自定义PyFunc数据增强，请使用如下参数：

      .. image:: map_parameter_pyfunc_cn.png

    参数：
        - **operations** (Union[list[TensorOperation], list[functions]]) - 一组数据增强操作，支持数据集增强操作或者用户自定义的Python Callable对象。map操作将按顺序将一组数据增强作用在数据集对象上。
        - **input_columns** (Union[str, list[str]], 可选) - 第一个数据增强操作的输入数据列。此列表的长度必须与 `operations` 列表中第一个数据增强的预期输入列数相匹配。默认值： ``None`` ，表示所有数据列都将传递给第一个数据增强操作。
        - **output_columns** (Union[str, list[str]], 可选) - 最后一个数据增强操作的输出数据列。如果 `input_columns` 长度不等于 `output_columns` 长度，则必须指定此参数。列表的长度必须与最后一个数据增强的输出列数相匹配。默认值： ``None`` ，输出列将与输入列具有相同的名称。
        - **num_parallel_workers** (int, 可选) - 指定map操作的多进程/多线程并发数，加快处理速度。默认值： ``None`` ，将使用 `set_num_parallel_workers` 设置的并发数。
        - **\*\*kwargs** - 其他参数。

          - python_multiprocessing (bool, 可选) - 启用Python多进程模式加速map操作。当传入的 `operations` 计算量很大时，开启此选项可能会有较好效果。默认值： ``False`` 。
          - max_rowsize (Union[int, list[int]], 可选) - 指定在多进程之间复制数据时，共享内存分配的基本单位，单位为MB。总占用的共享内存会随着 ``num_parallel_workers`` 和 :func:`mindspore.dataset.config.set_prefetch_size` 增加而变大。
            仅当 `python_multiprocessing` 为 ``True`` 时，该选项有效。默认值： ``None`` ，动态分配共享内存（后续版本将废弃此参数）。

            - 如果设置为 ``-1`` / ``None`` ，共享内存将随数据大小动态分配；
            - 如果是int值，代表 ``input_columns`` 和 ``output_columns`` 均使用该值为单位创建共享内存；
            - 如果是列表，第一个元素代表 ``input_columns`` 使用该值为单位创建共享内存，第二个元素代表 ``output_columns`` 使用该值为单位创建共享内存。
          - cache (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。
          - callbacks (DSCallback, list[DSCallback], 可选) - 要调用的Dataset回调函数列表。默认值： ``None`` 。
          - offload (bool, 可选) - 是否进行异构硬件加速，详情请阅读 `数据准备异构加速 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/dataset_offload.html>`_ 。默认值： ``None`` 。

    .. warning::
        在多进程 `spawn` 模式下， `map` 会隐式使用 `dill` 模块对 `operations` 进行序列化/反序列化，而该模块存在已知安全隐患。
        攻击者可构造恶意 pickle 数据，在反序列化过程中执行任意代码。切勿加载可能来自不可信来源或已被篡改的数据。

    .. note::
        - 参数 `max_rowsize` 将在后续版本废弃。
        - `operations` 参数接收 `TensorOperation` 类型的数据处理操作，以及用户定义的Python函数（PyFuncs）。
        - 通过ds.config.set_multiprocessing_start_method("spawn")方式设置多进程的启动方式为 `spawn` 模式，且 `python_multiprocessing=True`
          和 `num_parallel_workers>1` 时，支持将 `mindspore.nn` 和 `mindspore.ops` 或其他的网络计算算子添加到 `operations` 中，
          否则不支持添加到 `operations` 中。
        - 当前仅部分场景支持在 `operations` 参数传入的Python函数中调用DVPP算子：

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

    返回：
        Dataset，应用了上述操作的新数据集对象。
