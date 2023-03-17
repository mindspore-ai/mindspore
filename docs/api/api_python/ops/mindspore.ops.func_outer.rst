mindspore.ops.outer
====================

.. py:function:: mindspore.ops.outer(input, vec2)

    计算 `input` 和 `vec2` 的外积。如果向量 `input` 长度为 :math:`n` ， `vec2` 长度为 :math:`m` ，则输出矩阵shape为 :math:`(n, m)` 。

    .. note::
        该函数不支持广播。

    参数：
        - **input** (Tensor) - 输入一维向量。
        - **vec2** (Tensor) - 输入一维向量。

    返回：
        out (Tensor, optional)，两个一维向量的外积，是一个2维矩阵，。

    异常：
        - **TypeError** - 如果 `input` 或 `vec2` 不是Tensor。
        - **ValueError** - 如果 `input` 或 `vec2` 不是一维Tensor。
