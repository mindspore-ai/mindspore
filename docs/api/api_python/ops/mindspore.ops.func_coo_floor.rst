mindspore.ops.coo_floor
========================

.. py:function:: mindspore.ops.coo_floor(x)

    逐元素对COOTensor向下取整至最接近的整数。

    .. math::
        out_i = \lfloor x_i \rfloor

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (COOTensor) - Floor的输入，任意维度的COOTensor。其数据类型必须为float16、float32、float64。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64。
