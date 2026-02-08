mindspore.ops.csr_cosh
=======================

.. py:function:: mindspore.ops.csr_cosh(x)

    逐元素计算CSRTensor `x` 的双曲余弦值。

    .. math::
        out_i = \cosh(x_i)

    参数：
        - **x** (CSRTensor) - csr_cosh的输入，任意维度的CSRTensor，其数据类型为float16、float32、float64、complex64、complex128。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、complex64、complex128。
        - **TypeError** - `x` 不是CSRTensor。
