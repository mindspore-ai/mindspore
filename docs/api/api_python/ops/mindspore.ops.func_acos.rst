mindspore.ops.acos
===================

.. py:function:: mindspore.ops.acos(x)

    逐元素计算输入Tensor的反余弦。

    .. math::
        out_i = cos^{-1}(x_i)

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型应该是以下类型之一：float16、float32、float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16、float32或float64。
