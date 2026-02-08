mindspore.mint.cummin
======================

.. py:function:: mindspore.mint.cummin(input, dim)

    返回tensor在指定维度上的累积最小值及其索引。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \min(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    .. note::
        Ascend不支持GE后端。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定计算的维度。

    返回：
        两个tensor组成的tuple(min, min_indices)
