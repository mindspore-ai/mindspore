mindspore.ops.sin
==================

.. py:function:: mindspore.ops.sin(x)

    逐元素计算输入Tensor的正弦。

    .. math::
        out_i = sin(x_i)

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(N,*)` 其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
