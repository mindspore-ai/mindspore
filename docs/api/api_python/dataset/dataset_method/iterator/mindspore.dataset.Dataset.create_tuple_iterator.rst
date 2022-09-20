mindspore.dataset.Dataset.create_tuple_iterator
===============================================

.. py:method:: mindspore.dataset.Dataset.create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

    基于数据集对象创建迭代器。输出数据为 `numpy.ndarray` 组成的列表。

    可以通过参数 `columns` 指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

    参数：
        - **columns** (list[str], 可选) - 用于指定输出的数据列和列的顺序。默认值：None，输出所有数据列。
        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。
        - **do_copy** (bool, 可选) - 当参数 `output_numpy` 为False，即输出数据类型为mindspore.Tensor时，可以将此参数指定为False以减少拷贝，获得更好的性能。默认值：True。

    返回：
        TupleIterator，基于数据集对象创建的元组迭代器。
