mindspore.ops.reshape
======================

.. py:function:: mindspore.ops.reshape(input, shape)

    基于给定的 `shape` ，对输入Tensor进行重新排列。

    `shape` 最多只能有一个-1，在这种情况下，它可以从剩余的维度和输入的元素个数中推断出来。

    参数：
        - **input** (Tensor) - Tensor的 `shape` 为 :math:`(x_1, x_2, ..., x_R)` 。
        - **shape** (Union[tuple[int], list[int], Tensor[int]]) - 如果 `shape` 是list或者tuple，其元素需为整数，
          并且只支持常量值。 如 :math:`(y_1, y_2, ..., y_S)` 。如果 `shape` 是张量，则其数据类型为int32或者int64，并且只支持一维张量。

    返回：
        Tensor，若给定的 `shape` 中不包含-1, 则输出 `shape` 为 :math:`(y_1, y_2, ..., y_S)` 。若给定的 `shape` 中第 `k` 个位置
        为-1，则输出 `shape` 为 :math:`(y_1, ..., y_{k-1}, \frac{\prod_{i=1}^{R}x_{i}}{y_1\times ...\times y_{k-1}\times
        y_{k+1}\times...\times y_S} , y_{k+1},..., y_S)`。

    异常：
        - **ValueError** - 给定的 `shape` 中包含一个以上的-1。
        - **ValueError** - 给定的 `shape` 中包含小于-1的元素。
        - **ValueError** - 给定的 `shape` 中不包含-1的场景，各元素的乘积不等于输入Tensor的 `shape` 的乘积，
          :math:`\prod_{i=1}^{R}x_{i} \ne \prod_{i=1}^{S}y_{i}`，（即与输入的数组大小不匹配）。
          或者给定的 `shape` 中包含-1的场景，除-1外元素的乘积无法整除输入Tensor的 `shape` 的乘积 :math:`\prod_{i=1}^{R}x_{i}` 。
