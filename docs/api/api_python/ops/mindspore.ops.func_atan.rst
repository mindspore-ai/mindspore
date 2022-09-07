mindspore.ops.atan
===================

.. py:function:: mindspore.ops.atan(x)

    逐元素计算输入Tensor的反正切值。

    .. math::
        out_i = tan^{-1}(x_i)

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(N,*)` 其中 :math:`*` 表示任意数量的附加维度。数据类型支持：float16、float32。

    返回：
        Tensor的数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16或float32。
