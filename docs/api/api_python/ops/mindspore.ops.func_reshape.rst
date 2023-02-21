mindspore.ops.reshape
======================

.. py:function:: mindspore.ops.reshape(input, shape)

    基于给定的shape，对输入Tensor进行重新排列。

    `shape` 最多只能有一个-1，在这种情况下，它可以从剩余的维度和输入的元素个数中推断出来。

    参数：
        - **input** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **shape** (Union[tuple[int], Tensor[int]]) - 输入tuple由多个整数构成，如 :math:`(y_1, y_2, ..., y_S)` 。只支持常量值。

    返回：
        Tensor，其shape为 :math:`(y_1, y_2, ..., y_S)` 。

    异常：
        - **ValueError** - 给定的 `Fshape`，如果它有多个-1，或者除-1（若存在）之外的元素的乘积小于或等于0，或者无法被输入Tensor的shape的乘积整除，或者与输入的数组大小不匹配。
