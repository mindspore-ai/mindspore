mindspore.ops.Square
=====================

.. py:class:: mindspore.ops.Square

    逐元素计算输入Tensor的平方。

    .. math::
        out_{i} = (x_{i})^2

    输入：
        - **x** (Tensor) - 输入Tensor。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
