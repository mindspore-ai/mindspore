mindspore.mint.log2
===================

.. py:function:: mindspore.mint.log2(input)

    逐元素返回tensor以2为底的对数。

    .. math::
        y_i = \log_2(x_i)

    .. warning::
        如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    参数：
        - **input** (Tensor) - 任意维度的输入tensor。其值必须大于0。

    返回：
        Tensor，具有与 `input` 相同的shape。如果 `input.dtype` 为整数或布尔类型，输出数据类型为float32。否则输出数据类型与 `input.dtype` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。