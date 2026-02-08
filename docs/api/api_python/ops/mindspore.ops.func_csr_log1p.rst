mindspore.ops.csr_log1p
========================

.. py:function:: mindspore.ops.csr_log1p(x)

    对输入CSRTensor逐元素加一后计算自然对数。

    .. math::
        out_i = \text{log_e}(x_i + 1)

    参数：
        - **x** (CSRTensor) - 输入CSRTensor。数据类型为float16或float32。
          该值必须大于-1。

    返回：
        CSRTensor，与 `x` 的shape相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型非float16或float32。
