mindspore.ops.coo_sinh
=======================

.. py:function:: mindspore.ops.coo_sinh(x)

    逐元素计算输入COOTensor的双曲正弦。

    .. math::
        out_i = \sinh(x_i)

    参数：
        - **x** (COOTensor) - coo_sinh的输入COOTensor。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
