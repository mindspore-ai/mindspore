mindspore.ops.coo_cosh
=======================

.. py:function:: mindspore.ops.coo_cosh(x)

    逐元素计算COOTensor `x` 的双曲余弦值。

    .. math::
        out_i = \cosh(x_i)

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (COOTensor) - coo_cosh的输入，任意维度的COOTensor，其数据类型为float16、float32、float64、complex64、complex128。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **TypeError** - `x` 不是COOTensor。
