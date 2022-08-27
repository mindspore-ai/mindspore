mindspore.ops.atanh
====================

.. py:function:: mindspore.ops.atanh(x)

    逐元素计算输入Tensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
       实验性算子，将来有可能更改或删除。

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型支持：float16、float32。

    返回：
        Tensor的数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16或float32。
