mindspore.ops.linspace
======================

.. py:function:: mindspore.ops.linspace(start, stop, num)

    返回一个在区间 `start` 和 `stop` （包括 `start` 和 `stop` ）内均匀分布的，包含 `num` 个值的一维Tensor。

    .. math::
        \begin{aligned}
        &step = (stop - start)/(num - 1)\\
        &output = [start, start+step, start+2*step, ... , stop]
        \end{aligned}

    参数：
        - **start** (Union[Tensor, int, float]) - 零维Tensor，数据类型必须为float32。区间的起始值。
        - **stop** (Union[Tensor, int, float]) - 零维Tensor，数据类型必须为float32。区间的末尾值。
        - **num** (Union[Tensor, int]) - 间隔中的包含的数值数量，包括区间端点。必须为正数。

    返回：
        Tensor，具有与 `start` 相同的dtype，shape为 :math:`(num)` 。

    异常：
        - **TypeError** - `start` 或 `stop` 不是Tensor。
        - **TypeError** - `start` 或 `stop` 的数据类型不是float32。
        - **ValueError** - `start` 或 `stop` 的维度不是0。
        - **TypeError** - `num` 不是int类型。
        - **ValueError** - `num` 不是正数。
