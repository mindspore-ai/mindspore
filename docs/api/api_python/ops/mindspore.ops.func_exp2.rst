mindspore.ops.exp2
==================

.. py:function:: mindspore.ops.exp2(x)

    逐元素计算Tensor `x` 以2为底的指数。

    .. math::
        out_i = 2^{x_i}

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
