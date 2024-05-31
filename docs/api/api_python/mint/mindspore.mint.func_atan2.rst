mindspore.mint.atan2
====================

.. py:function:: mindspore.mint.atan2(input, other)

    逐元素计算input/other的反正切值。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`input = r*\sin(\theta), other = r*\cos(\theta)` ，其中 :math:`r = \sqrt{input^2 + other^2}` 。

    .. note::
        - 参数 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。如果两参数数据类型不一致，则低精度类型会被转换成较高精度类型。

    参数：
        - **input** (Tensor, Number.number) - 输入Tensor或常数。
        - **other** (Tensor, Number.number) - 输入Tensor或常数，shape与 `input` 相同，或能与 `input` 的shape广播。

    返回：
        Tensor，与广播后的输入shape相同，和 `input` 数据类型相同。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor或常数。
        - **RuntimeError** - `input` 与 `other` 之间的数据类型转换不被支持。
