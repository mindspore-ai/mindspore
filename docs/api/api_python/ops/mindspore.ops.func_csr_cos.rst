mindspore.ops.csr_cos
======================

.. py:function:: mindspore.ops.csr_cos(x)

    逐元素计算CSRTensor输入的余弦。

    .. math::
        out_i = \cos(x_i)

    .. warning::
        目前支持float16、float32数据类型。如果使用float64，可能会存在精度丢失的问题。

    参数：
        - **x** (CSRTensor) - CSRTensor的输入。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16、float32、float64、complex64或complex128。
