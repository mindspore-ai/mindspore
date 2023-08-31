mindspore.nn.piecewise_constant_lr
====================================

.. py:function:: mindspore.nn.piecewise_constant_lr(milestone, learning_rates)

    获取分段常量学习率。每个step的学习率将会被存放在一个列表中。

    通过给定的 `milestone` 和 `learning_rates` 计算学习率。设 `milestone` 的值为 :math:`(M_1, M_2, ..., M_t, ..., M_N)` ， `learning_rates` 的值为 :math:`(x_1, x_2, ..., x_t, ..., x_N)` 。N是 `milestone` 的长度。
    设 :math:`y` 为输出学习率，那么对于第 :math:`i` 步，计算 :math:`y[i]` 的公式为：

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    参数：
        - **milestone** (Union[list[int], tuple[int]]) - milestone列表。当达到指定的step时，使用对应的 `learning_rates`。 此列表是一个单调递增的列表。列表中的元素必须大于0。
        - **learning_rates** (Union[list[float], tuple[float]]) - 学习率列表。

    返回：
        list[float]。列表的大小为 :math:`M_N`。

    异常：
        - **TypeError** - `milestone` 或 `learning_rates` 既不是tuple也不是list。
        - **ValueError** - `milestone` 和 `learning_rates` 的长度不相等。
        - **ValueError** - `milestone` 中的不是单调递增的。
