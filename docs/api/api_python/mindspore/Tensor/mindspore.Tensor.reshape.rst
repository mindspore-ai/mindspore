mindspore.Tensor.reshape
========================

.. py:method:: mindspore.Tensor.reshape(*shape)

    基于给定的shape，对输入Tensor进行重新排列。

    `shape` 最多只能有一个-1，在这种情况下，它可以从剩余的维度和输入的元素个数中推断出来。

    参数：
        - **shape** (Union[int, tuple[int], list[int]]) - 如果`shape`是list或者tuple，其元素需为整数，并且只支持常量值。
            如 :math:`(y_1, y_2, ..., y_S)` 。

    返回：
        Tensor，若给定的`shape`中不包含-1, 则输出`shape`为 :math:`(y_1, y_2, ..., y_S)` 。若给定的`shape`中第`k`个位置为-1，
            则输出`shape`为 :math:`(y_1, ..., y_{k-1}, \frac{\prod_{i=1}^{R}x_{i}}{y_1\times ...\times y_{k-1}\times
            y_{k+1}\times...\times y_S} , y_{k+1},..., y_S)`。
