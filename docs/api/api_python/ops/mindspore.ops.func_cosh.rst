mindspore.ops.cosh
===================

.. py:function:: mindspore.ops.cosh(input)

    逐元素计算 `input` 的双曲余弦值。

    .. math::
        out_i = cosh(input_i)

    参数：
        - **input** (Tensor) - cosh的输入，任意维度的Tensor，其数据类型为float16、float32、float64、complex64、complex128。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、float64、complex64、complex128。
        - **TypeError** - `input` 不是Tensor。
