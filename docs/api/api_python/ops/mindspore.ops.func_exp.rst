mindspore.ops.exp
=================

.. py:function:: mindspore.ops.exp(input)

    逐元素计算 `input` 的指数。

    .. math::

        out_i = e^{x_i}

    参数：
        - **input** (Tensor) - 指数函数的输入Tensor。维度需要在 [0, 7] 的范围。

    返回：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - `input` 不是Tensor。
