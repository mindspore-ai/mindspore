mindspore.ops.csr_inv
======================

.. py:function:: mindspore.ops.csr_inv(x)

    逐元素计算输入CSRTensor的倒数。

    .. math::
        out_i = \frac{1}{x_{i} }

    参数：
        - **x** (CSRTensor) - 任意维度的CSRTensor。其数据类型为float16、float32或int32。

    返回：
        CSRTensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型不为float16、float32或int32。
