mindspore.ops.log
=================

.. py:function:: mindspore.ops.log(x)

    逐元素返回Tensor的自然对数。

    .. math::
        y_i = log_e(x_i)

    .. warning::
        如果算子Log的输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    .. note::
        Ascend上输入Tensor的维度要小于等于8，CPU上输入Tensor的维度要小于8。

    参数：
        - **x** (Tensor) - 任意维度的输入Tensor。该值必须大于0。

    返回：
        Tensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - 在CPU平台上运行时，`x` 的数据类型不是float16、float32或float64。
        - **TypeError** - 在Ascend平台上运行时，`x` 的数据类型不是float16或float32。
