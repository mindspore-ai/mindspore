mindspore.dataset.Dataset.create_tuple_iterator
===============================================

.. py:method:: mindspore.dataset.Dataset.create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

    创建数据集迭代器，返回列表形式的样本，其中的元素为各列数据。

    参数：
        - **columns** (list[str], 可选) - 指定输出数据列及其顺序。默认值： ``None`` ，保留所有输出列及其原始顺序。
        - **num_epochs** (int, 可选) - 数据集迭代次数。默认值： ``-1`` ，数据集可以无限迭代。
        - **output_numpy** (bool, 可选) - 是否保持输出数据类型为 NumPy 数组，否则转换为 :class:`mindspore.Tensor` 。默认值： ``False`` 。
        - **do_copy** (bool, 可选) - 指定转换输出类型为 :class:`mindspore.Tensor` 时是否拷贝数据，否则直接复用数据缓冲区以获得更好的性能，仅当 `output_numpy` 为 ``False`` 时有效。默认值： ``True`` 。

    返回：
        Iterator，返回列表形式样本的迭代器。
