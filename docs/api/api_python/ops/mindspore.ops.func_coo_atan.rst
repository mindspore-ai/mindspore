mindspore.ops.coo_atan
=======================

.. py:function:: mindspore.ops.coo_atan(x: COOTensor)

    逐元素计算输入COOTensor的反正切值。

    .. math::
        out_i = tan^{-1}(x_i)

    参数：
        - **x** (COOTensor) - COOTensor的输入，数据类型支持：float16、float32。

    返回：
        COOTensor的数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16或float32。
