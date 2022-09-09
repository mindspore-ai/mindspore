mindspore.ops.atan2
===================

.. py:function:: mindspore.ops.atan2(x, y)

    逐元素计算x/y的反正切值。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`x = r*\sin(\theta), y = r*\cos(\theta)` ，其中 :math:`r = \sqrt{x^2 + y^2}` 。
    输入 `x` 和 `y` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被转换到高精度的数据类型。

    参数：
        - **x** (Tensor) - 输入Tensor，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **y** (Tensor) - 输入Tensor，shape应能在广播后与 `x` 相同，或 `x` 的shape在广播后与 `y` 相同。

    返回：
        Tensor，与广播后的输入shape相同，和 `x` 数据类型相同。

    异常：
        - **TypeError** - `x` 或 `y` 不是Tensor。
        - **RuntimeError** - `x` 与 `y` 之间的数据类型转换不被支持
