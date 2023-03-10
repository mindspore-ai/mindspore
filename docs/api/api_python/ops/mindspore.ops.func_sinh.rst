mindspore.ops.sinh
===================

.. py:function:: mindspore.ops.sinh(input)

    逐元素计算输入Tensor的双曲正弦。

    .. math::
        out_i = \sinh(input_i)

    参数：
        - **input** (Tensor) - sinh的输入Tensor，其秩范围必须在[0, 7]。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
