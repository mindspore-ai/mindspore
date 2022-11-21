mindspore.ops.csr_sin
======================

.. py:function:: mindspore.ops.csr_sin(x: CSRTensor)

    逐元素计算输入CSRTensor的正弦。

    .. math::
        out_i = csr_sin(x_i)

    参数：
        - **x** (CSRTensor) - CSRTensor的shape为 :math:`(N,*)` 其中 :math:`*` 表示任意数量的附加维度。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
