mindspore.ops.coo_floor
========================

.. py:function:: mindspore.ops.coo_floor(x: COOTensor)

    COOTensor逐元素向下取整函数。

    .. math::
        out_i = \lfloor x_i \rfloor

    参数：
        - **x** (COOTensor) - Floor的输入，任意维度的COOTensor，秩应小于8。其数据类型必须为float16、float32。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是COOTensor。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64。