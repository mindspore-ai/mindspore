mindspore.ops.coo_log
======================

.. py:function:: mindspore.ops.coo_log(x: COOTensor)

    逐元素返回COOTensor的自然对数。

    .. math::
        y_i = log_e(x_i)

    .. warning::

        如果算子Log的输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    参数：
        - **x** (COOTensor) - 任意维度的输入COOTensor。该值必须大于0。

    返回：
        COOTensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - 在GPU和CPU平台上运行时，`x` 的数据类型不是float16、float32或float64。
        - **TypeError** - 在Ascend平台上运行时，`x` 的数据类型不是float16或float32。
