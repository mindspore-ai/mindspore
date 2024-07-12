mindspore.ops.cosh
===================

.. py:function:: mindspore.ops.cosh(input)

    逐元素计算 `input` 的双曲余弦值。

    .. math::
        out_i = \cosh(input_i)

    参数：
        - **input** (Tensor) - cosh的输入，任意维度的Tensor，支持数据类型：

          - GPU/CPU： float16、float32、float64、complex64或complex128。
          - Ascend： float16、float32、float64、complex64、complex128或bfloat16。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 

          - CPU/GPU: 如果 `input` 的数据类型不是float16、float32、float64、complex64或complex128。
          - Ascend: 如果 `input` 的数据类型不是float16、float32、float64、complex64、complex128或bfloat16。
