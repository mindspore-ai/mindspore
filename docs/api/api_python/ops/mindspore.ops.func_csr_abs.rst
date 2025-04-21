mindspore.ops.csr_abs
======================

.. py:function:: mindspore.ops.csr_abs(x: CSRTensor)

    逐元素计算输入CSRTensor的绝对值。

    .. math::
        out_i = |x_i|

    参数：
        - **x** (CSRTensor) - 输入CSRTensor。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
