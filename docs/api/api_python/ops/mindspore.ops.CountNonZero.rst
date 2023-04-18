mindspore.ops.CountNonZero
==========================

.. py:class:: mindspore.ops.CountNonZero(dims=None)

    计算输入Tensor指定轴上的非零元素的数量。

    更多详情请查看： :func:`mindspore.ops.count_nonzero` 。

    参数：
        - **dims** (Union[int, tuple(int), list(int)]，可选) - 要沿其计算非零值数量的维度。默认值：None，计算所有非零元素的个数。

    输入：
        - **x** (Tensor) - 要计算非零元素个数的输入。shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。

    输出：
        Tensor， `dims` 指定的轴上非零元素数量。
