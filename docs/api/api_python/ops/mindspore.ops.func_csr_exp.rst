mindspore.ops.csr_exp
======================

.. py:function:: mindspore.ops.csr_exp(x: CSRTensor)

    逐元素计算CSRTensor `x` 的指数。

    .. math::

        out_i = e^{x_i}

    参数：
        - **x** (CSRTensor) - 指数函数的输入CSRTensor。维度需要在 [0, 7] 的范围。

    返回：
        CSRTensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
