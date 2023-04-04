mindspore.ops.sqrt
==================

.. py:function:: mindspore.ops.sqrt(x)

    逐元素返回当前Tensor的平方根。

    .. math::
        out_{i} = \sqrt{x_{i}}

    参数：
        - **x** (Tensor) - 输入Tensor，数据类型为number.Number。

    返回：
        Tensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。

