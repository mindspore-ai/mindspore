mindspore.ops.cummax
====================

.. py:function:: mindspore.ops.cummax(input, axis)

    返回tensor在指定轴上的累积最大值及其索引。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算的轴。

    返回：
        两个tensor组成的tuple(max, max_indices)
