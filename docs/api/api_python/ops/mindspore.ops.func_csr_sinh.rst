mindspore.ops.csr_sinh
=======================

.. py:function:: mindspore.ops.csr_sinh(x)

    逐元素计算输入CSRTensor的双曲正弦。

    .. math::
        out_i = \sinh(x_i)

    参数：
        - **x** (CSRTensor) - csr_sinh的输入CSRTensor。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
