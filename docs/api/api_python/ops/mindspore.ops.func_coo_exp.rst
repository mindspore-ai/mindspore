mindspore.ops.coo_exp
======================

.. py:function:: mindspore.ops.coo_exp(x)

    逐元素计算COOTensor `x` 的指数。

    .. math::

        out_i = e^{x_i}

    参数：
        - **x** (COOTensor) - 指数函数的输入COOTensor。

    返回：
        COOTensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是COOTensor。
