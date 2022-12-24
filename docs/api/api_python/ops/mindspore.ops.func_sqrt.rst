mindspore.ops.sqrt
==================

.. py:function:: mindspore.ops.sqrt(x)

    逐元素返回当前Tensor的平方。

    .. math::
        y_i = \sqrt(x_i)

    参数：
        - **x** (Tensor) - 任意维度的输入Tensor。该值必须大于0。

    返回：
        Tensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
