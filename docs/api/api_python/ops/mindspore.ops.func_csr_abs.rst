mindspore.ops.csr_abs
======================

.. py:function:: mindspore.ops.csr_abs(x)

    逐元素计算输入CSRTensor的绝对值。

    .. math::
        out_i = |x_i|

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (CSRTensor) - 输入CSRTensor。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
