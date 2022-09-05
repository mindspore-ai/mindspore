mindspore.ops.exp
=================

.. py:function:: mindspore.ops.exp(x)

    逐元素计算 `x` 的指数。

    .. math::

        out_i = e^{x_i}

    参数：
        - **x** (Tensor) - 指数函数的输入Tensor。维度需要在 [0, 7] 的范围。

    返回：
        Tensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
