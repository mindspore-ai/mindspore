mindspore.ops.coo_abs
======================

.. py:function:: mindspore.ops.coo_abs(x)

    逐元素计算输入COOTensor的绝对值。

    .. math::
        out_i = |x_i|

    参数：
        - **x** (COOTensor) - 输入COOTensor。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
