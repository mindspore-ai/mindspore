mindspore.ops.csr_atan
=======================

.. py:function:: mindspore.ops.csr_atan(x)

    逐元素计算输入CSRTensor的反正切值。

    .. math::
        out_i = \tan^{-1}(x_i)

    参数：
        - **x** (CSRTensor) - 数据类型支持：float16、float32。

    返回：
        CSRTensor，数据类型与输入相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型不是float16或float32。
