mindspore.ops.csr_floor
========================

.. py:function:: mindspore.ops.csr_floor(x)

    CSRTensor逐元素向下取整函数。

    .. math::
        out_i = \lfloor x_i \rfloor

    参数：
        - **x** (CSRTensor) - Floor的输入，任意维度的CSRTensor。其数据类型必须为float16、float32或float64。

    返回：
        CSRTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的类型不是CSRTensor。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。