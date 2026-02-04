mindspore.ops.csr_expm1
========================

.. py:function:: mindspore.ops.csr_expm1(x)

    逐元素计算输入CSRTensor的指数，然后减去1。

    .. math::
        out_i = e^{x_i} - 1

    参数：
        - **x** (CSRTensor) - 数据类型为float16或float32的CSRTensor。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
