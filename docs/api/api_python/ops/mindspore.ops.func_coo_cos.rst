mindspore.ops.coo_cos
======================

.. py:function:: mindspore.ops.coo_cos(x)

    逐元素计算COOTensor输入的余弦。

    .. math::
        out_i = \cos(x_i)

    .. warning::
        目前支持float16、float32数据类型。如果使用float64，可能会存在精度丢失的问题。

    参数：
        - **x** (COOTensor) - COOTensor的输入。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、complex64或complex128。
