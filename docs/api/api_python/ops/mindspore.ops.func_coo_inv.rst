mindspore.ops.coo_inv
======================

.. py:function:: mindspore.ops.coo_inv(x)

    逐元素计算输入COOTensor的倒数。

    .. math::
        out_i = \frac{1}{x_{i} }

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (COOTensor) - 任意维度的COOTensor。其数据类型为float16、float32或int32。

    返回：
        COOTensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - `x` 的数据类型不为float16、float32或int32。
