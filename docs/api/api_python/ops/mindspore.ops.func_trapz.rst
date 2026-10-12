mindspore.ops.trapz
====================

.. py:function:: mindspore.ops.trapz(y, x=None, *, dx=1.0, dim=-1)

    计算沿指定维度的梯形法则积分。
    采样点之间的距离由tensor `x` 或标量 `dx` 指定，默认 ``1`` 。

    .. math::
        \mathop{ \int }\nolimits_{{}}^{{}}{y}{ \left( {x} \right) } \text{d} x

    参数：
        - **y** (Tensor) - 用于积分计算的输入tensor。
        - **x** (Tensor，可选) - 如果指定，则定义采样点之间的间距。

    关键字参数：
        - **dx** (float，可选) - 采样点之间的常数间距，默认 ``1.0`` 。如果 `x` 被指定，则 `dx` 不生效。
        - **dim** (int，可选) - 指定维度。默认 ``-1`` 。

    返回：
        Tensor