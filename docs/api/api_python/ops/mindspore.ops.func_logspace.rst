mindspore.ops.logspace
======================

.. py:function:: mindspore.ops.logspace(start, end, steps, base=10, *, dtype=mstype.float32)

    返回一个大小为 `steps` 的1-D Tensor，其值从 :math:`base^{start}` 到 :math:`base^{end}` ，以 `base` 为底数。

    .. note::
        - 输入 `base` 必须是整数。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    参数：
        - **start** (Union[float, Tensor]) - 间隔的起始值。
        - **end** (Union[float, Tensor]) - 间隔的结束值。
        - **steps** (int) - `steps` 必须为非负整数。
        - **base** (int，可选) - `base` 必须为非负整数。默认值：10。
        - **dtype** (mindspore.dtype，可选) - 输出的数据类型。默认值：mstype.float32。

    返回：
        Tensor，shape为 :math:`(step, )` ，数据类型由属性 `dtype` 设置。

    异常：
        - **TypeError** - 若 `start` 不是一个float或Tensor。
        - **TypeError** - 若 `end` 不是一个float或Tensor。
        - **TypeError** - 若 `steps` 不是一个整数。
        - **TypeError** - 若 `base` 不是一个整数。
        - **ValueError** - 若 `steps` 不是非负整数。
        - **ValueError** - 若 `base` 不是非负整数。