mindspore.mint.outer
====================

.. py:function:: mindspore.mint.outer(input, vec2)

    计算两个tensor的外积。如果向量 `input` 长度为 :math:`n` ， `vec2` 长度为 :math:`m` ，则输出矩阵shape为 :math:`(n, m)` 。

    .. note::
        该函数不支持广播。

    参数：
        - **input** (Tensor) - 一维输入tensor。
        - **vec2** (Tensor) - 一维输入tensor。

    返回：
        Tensor