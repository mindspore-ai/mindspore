mindspore.ops.atan2
===================

.. py:function:: mindspore.ops.atan2(input, other)

    逐元素计算input/other的反正切值。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`input = r*\sin(\theta), other = r*\cos(\theta)` ，其中 :math:`r = \sqrt{input^2 + other^2}` 。
    输入 `input` 和 `other` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被转换到高精度的数据类型。

    参数：
        - **input** (Tensor, Number.number) - 输入Tensor或常数，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **other** (Tensor, Number.number) - 输入Tensor或常数，shape应能在广播后与 `input` 相同，或 `input` 的shape在广播后与 `other` 相同。

        .. note::
            两个参数中，至少有一个需要为Tensor。

    返回：
        Tensor或常数，与广播后的输入shape相同，和 `input` 数据类型相同。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor或常数。
        - **RuntimeError** - `input` 与 `other` 之间的数据类型转换不被支持。
