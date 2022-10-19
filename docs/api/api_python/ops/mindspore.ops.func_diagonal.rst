mindspore.ops.diagonal
=======================

.. py:function:: mindspore.ops.diagonal(input, offset=0, dim1=0, dim2=1)

    返回 `input` 的部分视图，其相对于 `dim1` 和 `dim2` 的对角线元素作为维度附加在形状的末尾。

    参数：
        - **input** (Tensor) - 输入Tensor，其维度至少为2。
        - **offset** (int, 可选) - 对角线与主对角线的偏移。可以是正值或负值。默认值：0。
        - **dim1** (int, 可选) - 二维子数组的第一轴，对角线应该从这里开始。默认值：0。
        - **dim2** (int, 可选) - 二维子数组的第二轴，对角线应该从这里开始。默认值：1。

    返回：
        Tensor，如果Tensor是二维，则返回值是一维数组。

    异常：
        - **ValueError** - 输入Tensor的维度少于2。
