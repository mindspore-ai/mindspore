mindspore.ops.linspace
======================

.. py:function:: mindspore.ops.linspace(start, end, steps)

    返回一个在区间 `start` 和 `end` （包括 `start` 和 `end` ）内均匀分布的，包含 `steps` 个值的一维Tensor。

    .. math::
        \begin{aligned}
        &step_len = (end - start)/(steps - 1)\\
        &output = [start, start+step_len, start+2*step_len, ... , end]
        \end{aligned}

    参数：
        - **start** (Union[Tensor, int, float]) - 零维Tensor，数据类型必须为float32。区间的起始值。
        - **end** (Union[Tensor, int, float]) - 零维Tensor，数据类型必须为float32。区间的末尾值。
        - **steps** (Union[Tensor, int]) - 间隔中的包含的数值数量，包括区间端点。必须为正数。

    返回：
        Tensor，具有与 `start` 相同的dtype，shape为 :math:`(steps)` 。

    异常：
        - **TypeError** - `start` 或 `end` 不是Tensor。
        - **TypeError** - `start` 或 `end` 的数据类型不是float32。
        - **ValueError** - `start` 或 `end` 的维度不是0。
        - **TypeError** - `steps` 不是int类型。
        - **ValueError** - `steps` 不是正数。
