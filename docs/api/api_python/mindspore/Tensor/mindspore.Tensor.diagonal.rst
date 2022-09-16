mindspore.Tensor.diagonal
=========================

.. py:method:: mindspore.Tensor.diagonal(offset=0, axis1=0, axis2=1)

    返回指定的对角线。

    参数：
        - **offset** (int, 可选) - 对角线与主对角线的偏移。可以是正值或负值。默认为主对角线。
        - **axis1** (int, 可选) - 二维子数组的第一轴，对角线应该从这里开始。默认为第一轴(0)。
        - **axis2** (int, 可选) - 二维子数组的第二轴，对角线应该从这里开始。默认为第二轴。

    返回：
        Tensor，如果Tensor是二维，则返回值是一维数组。

    异常：
        - **ValueError** - 输入Tensor的维度少于2。