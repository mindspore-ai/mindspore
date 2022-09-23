.. py:method:: map(operations, input_columns=None, output_columns=None, column_order=None, num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16, offload=None)

    给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。

    每个数据增强操作将数据集对象中的一个或多个数据列作为输入，将数据增强的结果输出为一个或多个数据列。
    第一个数据增强操作将 `input_columns` 中指定的列作为输入。
    如果数据增强列表中存在多个数据增强操作，则上一个数据增强的输出列将作为下一个数据增强的输入列。

    最后一个数据增强的输出列的列名由 `output_columns` 指定，如果没有指定 `output_columns` ，输出列名与 `input_columns` 一致。

    参数：
        - **operations** (Union[list[TensorOperation], list[functions]]) - 一组数据增强操作，支持数据集增强算子或者用户自定义的Python Callable对象。map操作将按顺序将一组数据增强作用在数据集对象上。
        - **input_columns** (Union[str, list[str]], 可选) - 第一个数据增强操作的输入数据列。此列表的长度必须与 `operations` 列表中第一个数据增强的预期输入列数相匹配。默认值：None。表示所有数据列都将传递给第一个数据增强操作。
        - **output_columns** (Union[str, list[str]], 可选) - 最后一个数据增强操作的输出数据列。如果 `input_columns` 长度不等于 `output_columns` 长度，则必须指定此参数。列表的长度必须必须与最后一个数据增强的输出列数相匹配。默认值：None，输出列将与输入列具有相同的名称。
        - **column_order** (Union[str, list[str]], 可选) - 指定传递到下一个数据集操作的数据列的顺序。如果 `input_columns` 长度不等于 `output_columns` 长度，则必须指定此参数。注意：参数的列名不限定在 `input_columns` 和 `output_columns` 中指定的列，也可以是上一个操作输出的未被处理的数据列。默认值：None，按照原输入顺序排列。
        - **num_parallel_workers** (int, 可选) - 指定map操作的多进程/多线程并发数，加快处理速度。默认值：None，将使用 `set_num_parallel_workers` 设置的并发数。
        - **python_multiprocessing** (bool, 可选) - 启用Python多进程模式加速map操作。当传入的 `operations` 计算量很大时，开启此选项可能会有较好效果。默认值：False。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **callbacks** (DSCallback, list[DSCallback], 可选) - 要调用的Dataset回调函数列表。默认值：None。
        - **max_rowsize** (int, 可选) - 指定在多进程之间复制数据时，共享内存分配的最大空间，仅当 `python_multiprocessing` 为True时，该选项有效。默认值：16，单位为MB。
        - **offload** (bool, 可选) - 是否进行异构硬件加速，详情请阅读 `数据准备异构加速 <https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/dataset/dataset_offload.html>`_ 。默认值：None。

    .. note::
        - `operations` 参数接收 `TensorOperation` 类型的数据处理操作，以及用户定义的Python函数(PyFuncs)。
        - 不要将 `mindspore.nn` 和 `mindspore.ops` 或其他的网络计算算子添加到 `operations` 中。

    返回：
        MapDataset，map操作后的数据集。

.. py:method:: num_classes()

    获取数据集对象中所有样本的类别数目。

    返回：
        int，类别的数目。

.. py:method:: output_shapes(estimate=False)

    获取数据集对象中每列数据的shape。

    参数：
        - **estimate** (bool) - 如果 `estimate` 为 False，将返回数据集第一条数据的shape。
          否则将遍历整个数据集以获取数据集的真实shape信息，其中动态变化的维度将被标记为None（可用于动态shape数据集场景），默认值：False。

    返回：
        list，每列数据的shape列表。

.. py:method:: output_types()

    获取数据集对象中每列数据的数据类型。

    返回：
        list，每列数据的数据类型列表。
