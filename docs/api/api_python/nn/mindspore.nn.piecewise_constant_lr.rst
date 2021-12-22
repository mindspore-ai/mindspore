mindspore.nn.piecewise_constant_lr
====================================

.. py:class:: mindspore.nn.piecewise_constant_lr(milestone, learning_rates)

    获取分段常量学习率。

    通过给定的 `milestone` 和 `learning_rates` 计算学习率。设 `milestone` 的值为 :math:`(M_1, M_2, ..., M_t, ..., M_N)` ， `learning_rates` 的值为 :math:`(x_1, x_2, ..., x_t, ..., x_N)` 。N是 `milestone` 的长度。
    设 `y` 为输出学习率， 那么对于第i步，计算y[i]的公式为：

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    **参数：**

    - **milestone** (Union[list[int], tuple[int]]) - milestone列表。此列表是一个单调递增的列表。类表中的元素必须大于0。
    - **learning_rates** (Union[list[float], tuple[float]]) - 学习率列表。

    **返回：**

    list[float]。列表的大小为 :math:`M_N`。

    **样例：**
    >>> import mindspore.nn as nn
    >>>
    >>> milestone = [2, 5, 10]
    >>> learning_rates = [0.1, 0.05, 0.01]
    >>> output = nn.piecewise_constant_lr(milestone, learning_rates)
    >>> print(output)
    [0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
