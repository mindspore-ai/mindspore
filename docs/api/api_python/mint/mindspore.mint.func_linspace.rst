mindspore.mint.linspace
=======================

.. py:function:: mindspore.mint.linspace(start, end, steps, *, dtype=None)

    返回一个在区间 `start` 和 `end` （包括 `start` 和 `end` ）内均匀分布的，包含 `steps` 个值的一维Tensor。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    参数：
        - **start** (Union[float, int]) - 区间的起始值。可以为int或float。
        - **end** (Union[float, int]) - 区间的末尾值。可以为int或float。
        - **steps** (int) - 间隔中的包含的数值数量，包括区间端点。必须为正整数。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 期望输出Tensor的类型。默认值： ``None`` ，则输出类型为float32。

    返回：
        Tensor，具有与 `start` 相同的dtype，shape为 :math:`(steps)` ，数据类型由 `dtype` 指定。

    异常：
        - **TypeError** - `start` 或 `end` 的数据类型不支持。
        - **ValueError** - `steps` 不是正整数。
