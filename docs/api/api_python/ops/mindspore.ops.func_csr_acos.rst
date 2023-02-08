mindspore.ops.csr_acos
=======================

.. py:function:: mindspore.ops.csr_acos(x: CSRTensor)

    逐元素计算输入CSRTensor的反余弦。

    .. math::
        out_i = cos^{-1}(x_i)

    参数：
        - **x** (CSRTensor) - 输入CSRTensor。

    返回：
        CSRTensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16、float32或float64。
