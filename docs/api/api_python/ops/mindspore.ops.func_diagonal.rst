mindspore.ops.diagonal
=======================

.. py:function:: mindspore.ops.diagonal(input, offset=0, dim1=0, dim2=1)

    返回 `input` 特定的对角线视图。如果 `input` 为2-D，返回偏移量为 `offset` 的对角线元素。如果 `input` 维度大于2，则返回由 `dim1` 和 `dim2` 确定的2-D子数组的对角线视图。这种情况下，移除 `input` 的 `dim1` 和 `dim2` 维度，并且由 `dim1` 和 `dim2` 确定的对角线元素插入 `input` 的最后一维。

    参数：
        - **input** (Tensor) - 输入Tensor，其维度至少为2。
        - **offset** (int, 可选) - 对角线与主对角线的偏移。可以是正值或负值。默认值：0。
        - **dim1** (int, 可选) - 二维子数组的第一轴，对角线应该从这里开始。默认值：0。
        - **dim2** (int, 可选) - 二维子数组的第二轴，对角线应该从这里开始。默认值：1。

    返回：
        Tensor，如果Tensor是二维，则返回值是一维数组，如果输入维度大于2，则先移除维度 `dim1` 和 `dim2`， 然后在末尾插入新的一维来对应对角元素。

    异常：
        - **ValueError** - 输入Tensor的维度少于2。
