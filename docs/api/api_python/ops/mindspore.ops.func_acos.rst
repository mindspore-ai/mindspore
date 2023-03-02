mindspore.ops.acos
===================

.. py:function:: mindspore.ops.acos(input)

    逐元素计算输入Tensor的反余弦。

    .. math::
        out_i = cos^{-1}(input_i)

    参数：
        - **input** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不是float16、float32、float64、complex64或complex128。
