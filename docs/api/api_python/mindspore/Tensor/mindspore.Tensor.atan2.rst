mindspore.Tensor.atan2
======================

.. py:method:: mindspore.Tensor.atan2(y)

    逐元素计算x/y的反正切值。

    `x` 指的当前 Tensor。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`x = r*\sin(\theta), y = r*\cos(\theta)` ，其中 :math:`r = \sqrt{x^2 + y^2}` 。
    输入 `x` 和 `y` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被转换到高精度的数据类型。

    参数：
        - **y** (Tensor) - 输入Tensor。shape应能在广播后与 `x` 相同，或 `x` 的shape在广播后与 `y` 相同。

    返回：
        Tensor，与广播后的输入shape相同，和 `x` 数据类型相同。

    异常：
        - **TypeError** - `x` 或 `y` 不是Tensor。
        - **RuntimeError** - `x` 与 `y` 之间的数据类型转换不被支持。