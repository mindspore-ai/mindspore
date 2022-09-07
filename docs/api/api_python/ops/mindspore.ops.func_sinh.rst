mindspore.ops.sinh
===================

.. py:function:: mindspore.ops.sinh(x)

    逐元素计算输入Tensor的双曲正弦。

    .. math::
        out_i = \sinh(x_i)

    参数：
        - **x** (Tensor) - sinh的输入Tensor，其秩范围必须在[0, 7]。

    返回：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
