mindspore.ops.SmoothL1Loss
============================

.. py:class:: mindspore.ops.SmoothL1Loss(beta=1.0, reduction='none')

    计算平滑L1损失，该L1损失函数有稳健性。

    .. warning::
        该API在CPU上性能较差，推荐在Ascend/GPU上运行。

    更多参考详见 :func:`mindspore.ops.smooth_l1_loss`。

    参数：
        - **beta** (number，可选) - 控制损失函数在L1损失和L2损失间变换的阈值，默认值： ``1.0`` 。

          - Ascend：该值必须大于等于0。
          - CPU/GPU：该值必须大于0。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'none'`` 。

          - ``"none"``：不应用归约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 任意维度输入Tensor。支持数据类型：

          - Ascend：float16、float32、bfloat16。
          - CPU/GPU：float16、float32、float64。
        - **labels** (Tensor) - 真实值。

          - CPU/Ascend: 与 `logits` 的shape相同， `logits` 和 `labels` 遵循隐式类型转换规则，使数据类型一致。
          - GPU: 与 `logits` 的shape和数据类型相同。

    输出：
        Tensor，如果 `reduction` 为 ``'none'``，则输出Tensor的shape与 `logits` 的shape相同，否则shape为 :math:`()`。
