mindspore.ops.tan
===================

.. py:function:: mindspore.ops.tan(x)

    计算输入元素的正切值。

    .. math::
        out_i = tan(x_i)

    参数：
        - **x** (Tensor) - Tan的输入，任意维度的Tensor。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
