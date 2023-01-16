mindspore.ops.outer
====================

.. py:function:: mindspore.ops.outer(x1, x2)

    计算 `x1` 和 `x2` 的外积。如果向量 `x1` 长度为n， `x2` 长度为m，则输出矩阵尺寸为n x m。

    .. note::
        该函数不支持广播。

    参数：
        - **x1** (Tensor) - 输入一维向量。
        - **x2** (Tensor) - 输入一维向量。

    返回：
        out (Tensor, optional)，两个一维向量的外积，是一个2维矩阵，。

    异常：
        - **TypeError** - 如果 `x1` 或 `x2` 不是Tensor。
        - **ValueError** - 如果 `x1` 或 `x2` 不是一维Tensor。
