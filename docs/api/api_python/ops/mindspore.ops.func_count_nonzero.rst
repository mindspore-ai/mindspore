mindspore.ops.count_nonzero
============================

.. py:function:: mindspore.ops.count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32)

    计算输入Tensor指定轴上的非零元素的数量。如果没有指定维度，则计算Tensor中所有非零元素的数量。

    参数：
        - **x** (Tensor) - 要计算非零元素个数的输入。shape为 :math:`(*)` ，其中 :math:`*` 为任意维度。
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 要沿其计算非零值数量的维度。注意，其中对应轴的多个元素不能指向同一个轴。默认值： ``()`` ，计算所有非零元素的个数。
        - **keep_dims** (bool, 可选) - 是否保留 `axis` 指定的维度。如果为 ``True`` ，保留对应维度size为1，如果为 ``False`` ，不保留对应维度。默认值： ``False`` 。
        - **dtype** (Union[Number, mindspore.bool\_]，可选) - 输出Tensor的数据类型。默认值： ``mstype.int32`` 。

    返回：
        Tensor， `axis` 指定的轴上非零元素数量。 数据类型由 `dtype` 指定。

    异常：
        - **TypeError** - `axis` 不是int、tuple或者list。
        - **ValueError** - 如果 `aixs` 中的任何值不在 [-x_dims, x_dims) 范围内。