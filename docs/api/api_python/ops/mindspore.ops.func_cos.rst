mindspore.ops.cos
==================

.. py:function:: mindspore.ops.cos(x)

    逐元素计算输入的余弦。

    .. math::
        out_i = cos(x_i)

    .. warning::
        目前支持float16、float32数据类型。如果使用float64，可能会存在精度丢失的问题。

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
