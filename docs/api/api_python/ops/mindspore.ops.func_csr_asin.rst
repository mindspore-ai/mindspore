mindspore.ops.csr_asin
=======================

.. py:function:: mindspore.ops.csr_asin(x)

    逐元素计算输入CSRTensor的反正弦。

    .. math::
        out_i = \sin^{-1}(x_i)

    参数：
        - **x** (CSRTensor) - CSRTensor的输入。数据类型应该是以下类型之一：float16、float32、float64。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64。
