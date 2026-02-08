mindspore.ops.csr_sin
======================

.. py:function:: mindspore.ops.csr_sin(x)

    逐元素计算输入CSRTensor的正弦。

    .. math::
        out_i = \sin(x_i)

    参数：
        - **x** (CSRTensor) - CSRTensor的输入。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16、float32或者float64、complex64、complex128。
