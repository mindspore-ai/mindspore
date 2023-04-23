mindspore.ops.trapz
====================

.. py:function:: mindspore.ops.trapz(y, x=None, *, dx=1.0, dim=-1)

    使用梯形法则沿给定轴 `dim` 对 `y(x)` 进行积分。
    默认情况下，元素之间的 `x` 轴距离将被设定为1，或者它们可以由数组 `x` 或者标量 `dx` 提供。

    .. math::
        \mathop{ \int }\nolimits_{{}}^{{}}{y}{ \left( {x} \right) } \text{d} x

    参数：
        - **y** (Tensor) - 要积分的Tensor。
        - **x** (Tensor，可选) - 对应于 `y` 值的样本点。如果 `x` 为 ``None`` ，则采样点均匀间隔 `dx` ，默认值： ``None`` 。如果 `x` 不为 ``None`` ，则由 `dim` 指定的轴减去1后， `x` 的形状应与 `y` 相同或者可以广播到 `y` 。

    关键字参数：
        - **dx** (float，可选) - 当 `x` 为None时，采样点之间的间距。如果 `x` 被指定，则 `dx` 不生效。默认值： ``1.0`` 。
        - **dim** (int，可选) - 进行积分的维度。默认值： ``-1`` 。

    返回：
        浮点类型Tenor，用梯形规则近似的定积分。如果 `y` 是一维数组，则结果是浮点数。如果 `y` 是n维数组，结果是一个N-1维数组。

    异常：
        - **RuntimeError** - 如果 `x` 维度为1且x.shape[0]不等于y.shape[dim]。
        - **ValueError** - 如果 `dim` 不在 :math:`[-y.ndim, y.ndim)` 范围内。
        - **TypeError** - 如果 `y` 不是Tensor。
        - **TypeError** - 如果 `x` 不为None的同时不是Tensor。
        - **TypeError** - 如果 `dx` 不是浮点数。
        - **TypeError** - 如果 `dim` 不是整数。
