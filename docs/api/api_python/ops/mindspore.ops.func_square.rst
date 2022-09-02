mindspore.ops.square
====================

.. py:function:: mindspore.ops.square(x)

    逐元素返回Tensor的平方。

    .. math::
        y_i = x_i ^ 2

    参数：
        - **x** (Tensor) - 输入Tensor的维度范围为[0,7]，类型为数值类型。

    返回：
        Tensor，具有与当前Tensor相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
