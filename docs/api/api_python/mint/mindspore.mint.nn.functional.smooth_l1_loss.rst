mindspore.mint.nn.functional.smooth_l1_loss
===========================================

.. py:function:: mindspore.mint.nn.functional.smooth_l1_loss(input, target, reduction='mean', beta=1.0)

    计算平滑L1损失，该L1损失函数有稳健性。

    平滑L1损失是一种类似于MSELoss的损失函数，但对异常值相对不敏感，可以参阅论文 `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ 。

    给定长度为 :math:`N` 的两个输入 :math:`x,\  y` ，平滑L1损失的计算如下：

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 * \text{beta}, & \text{otherwise. }
        \end{cases}

    当 `reduction` 不是设定为 `none` 时，计算如下：

    .. math::
        L =
        \begin{cases}
            \operatorname{mean}(L_{i}), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L_{i}),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    其中， :math:`\text{beta}` 控制损失函数在线性与二次间变换的阈值， :math:`\text{beta} \geq 0` ，默认值是 ``1.0`` 。 :math:`N` 为batch size。

    .. note::
        参数 `input` 和 `target` 遵循隐式类型转换规则，使数据类型保持一致。如果两参数数据类型不一致，则低精度类型会被转换成较高精度类型。

    参数：
        - **input** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。支持数据类型：

          - Ascend：float16、float32、bfloat16。
        - **target** (Tensor) - shape： :math:`(N, *)` ，与 `input` 的shape相同。支持数据类型：

          - Ascend：float16、float32、bfloat16。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用归约方法。
          - ``'mean'``：计算输出元素的平均值。
          - ``'sum'``：计算输出元素的总和。
        - **beta** (number，可选) - 控制损失函数在L1Loss和L2Loss间变换的阈值，该值必须大于等于0。默认值： ``1.0`` 。

    返回：
        Tensor，数据类型与 `input` 相同。如果 `reduction` 为 ``'none'`` ，则输出为Tensor且与 `input` 的shape相同。否则shape为 :math:`()`。

    异常：
        - **TypeError** - `input` 或 `target` 不是 Tensor。
        - **RuntimeError** - `input` 或 `target` 的数据类型不是float16，float32和bfloat16中的任一者。
        - **ValueError** - `input` 与 `target` 的shape不同。
        - **ValueError** - `reduction` 不是 ``'none'`` ， ``'mean'`` 和 ``'sum'`` 中的任一者。
        - **TypeError** - `beta` 不是float，bool或int。
        - **RuntimeError** - `beta` 小于0。
