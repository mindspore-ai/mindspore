mindspore.ops.BinaryCrossEntropy
=================================

.. py:class:: mindspore.ops.BinaryCrossEntropy(reduction='mean')

    计算目标值和预测值之间的二值交叉熵损失值。

    将 `logits` 设置为 :math:`x` ， `labels` 设置为 :math:`y` ，输出为 :math:`\ell(x, y)` 。则，

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    其中， :math:`L` 表示所有批次的损失， :math:`l` 表示一个批次的损失，n表示1-N范围内的一个批次，:math:`w_n` 表示第 :math:`n` 批二进制交叉熵的权重。则，

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. warning::
        - :math:`x` 的值必须在0到1之间。

    参数：
        - **reduction** (str) - 指定输出的计算方式。取值为'none'、'mean'或'sum'。默认值：'mean'。

    输入：
        - **logits** (Tensor) - 输入预测值。任意维度的Tensor，其数据类型必须为float16或float32。
        - **labels** (Tensor) - 输入目标值，其shape和数据类型与 `logits` 相同。
        - **weight** (Tensor, 可选) - 每个批次二值交叉熵的权重。且shape和数据类型必须与 `logits` 相同。默认值：None。

    输出：
        Tensor，与 `logits` 有相同的数据类型。如果 `reduction` 为'none'，则shape与 `logits` 相同。否则，输出为Scalar Tensor。

    异常：
        - **TypeError** - `logits` 、 `labels` 及 `weight` 的数据类型既不是float16，也不是float32。
        - **ValueError** - `reduction` 不为'none'、'mean'或'sum'。
        - **ValueError** - `labels` 的shape与 `logits` 或  `weight` 不同。
        - **TypeError** - `logits` 、 `labels` 或 `weight` 不是Tensor。
