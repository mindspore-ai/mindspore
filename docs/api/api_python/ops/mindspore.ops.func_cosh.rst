mindspore.ops.cosh
===================

.. py:function:: mindspore.ops.cosh(x)

    逐元素计算 `x` 的双曲余弦值。

    .. math::
        out_i = cosh(x_i)

    参数：
        - **x** (Tensor) - cosh的输入，任意维度的Tensor，其数据类型为float16、float32、float64、complex64、complex128。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、complex64、complex128。
        - **TypeError** - `x` 不是Tensor。
