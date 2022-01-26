mindspore.nn.SmoothL1Loss
============================

.. py:class:: mindspore.nn.SmoothL1Loss(beta=1.0)

    SmoothL1损失函数，如果预测值和目标值的逐个元素绝对误差小于设定阈值 `beta` 则用平方项，否则用绝对误差项。

    给定两个输入 :math:`x,\  y`，SmoothL1Loss定义如下：

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\beta}, & \text{if } |x_i - y_i| < {\beta} \\
        |x_i - y_i| - 0.5 {\beta}, & \text{otherwise.}
        \end{cases}

    其中，:math:`{\beta}` 代表阈值 `beta` 。

    .. note::
        - SmoothL1Loss可以看成 :class:`mindspore.nn.L1Loss` 的修改版本，也可以看成 :class:`mindspore.nn.L1Loss` 和 :class:`mindspore.ops.L2Loss` 的组合。 
        - :class:`mindspore.nn.L1Loss` 计算两个输入Tensor之间的绝对误差，而 :class:`mindspore.ops.L2Loss` 计算两个输入Tensor之间的平方误差。 
        - :class:`mindspore.ops.L2Loss` 通常更快收敛，但对离群值的鲁棒性较差。该损失函数具有较好的鲁棒性。

    **参数：**

    **beta** (float) - 损失函数计算在L1Loss和L2Loss间变换的阈值。默认值：1.0。

    **输入：**

    - **logits** (Tensor) - 预测值，任意维度Tensor。数据类型必须为float16或float32。
    - **labels** (Tensor) - 目标值，数据类型和shape与 `logits` 相同的Tensor。

    **输出：**

    Tensor，数据类型和shape与 `logits` 相同。

    **异常：**

    - **TypeError** - `beta` 不是float。
    - **TypeError** - `logits` 或 `labels` 不是Tensor。
    - **TypeError** - `logits` 或 `labels` 的数据类型既不是float16，也不是float32。
    - **TypeError** - `logits` 的数据类型与 `labels` 不同。
    - **ValueError** - `beta` 小于或等于0。
    - **ValueError** - `logits` 的shape与 `labels` 不同。
