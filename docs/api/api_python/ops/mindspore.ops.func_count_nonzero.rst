mindspore.ops.count_nonzero
============================

.. py:function:: mindspore.ops.count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32)

    计算输入Tensor指定轴上的非零元素的数量。

    参数：
        - **x** (Tensor) - 要计算非零元素个数的输入。shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 要沿其计算非零值数量的维度。默认值：()，计算所有非零元素的个数。
        - **keep_dims** (bool, 可选) - 是否保留 `axis` 指定的维度。如果为True，保留对应维度size为1，如果为False，不保留对应维度。默认值：False。
        - **dtype** (Union[Number, mindspore.bool\_]，可选) - 输出Tensor的数据类型。默认值：mindspore.int32。

    返回：
        Tensor， `axis` 指定的轴上非零元素数量。 数据类型由 `dtype` 指定。

    异常：
        - **TypeError** - `axis` 不是int、tuple或者list。
        - **ValueError** - 如果 `aixs` 中的任何值不在 [-x_dims, x_dims) 范围内。
