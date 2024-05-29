mindspore.mint.log
==================

.. py:function:: mindspore.mint.log(input)

    逐元素返回Tensor的自然对数。

    .. math::
        y_i = \log_e(x_i)

    .. warning::
        如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    参数：
        - **input** (Tensor) - 任意维度的输入Tensor。其值必须大于0。

    返回：
        Tensor，具有与 `input` 相同的shape和数据类型。

    异常：
        - **TypeError** - `input` 不是Tensor。
