mindspore.nn.BCELoss
====================

.. py:class:: mindspore.nn.BCELoss(weight=None, reduction='mean')

    计算目标值和预测值之间的二值交叉熵损失值。

    将预测值设置为 :math:`x` ，目标值设置为 :math:`y` ，输出损失设置为 :math:`\ell(x, y)` 。

    则公式如下：

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    其中N是批次大小。公式如下：

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. note::
        预测值一般是sigmoid函数的输出。因为是二分类，所以目标值应是0或者1。如果输入是0或1，则上述损失函数是无意义的。

    参数：
        - **weight** (Tensor, 可选) - 指定每个批次二值交叉熵的权重。与输入数据的shape和数据类型相同。默认值：None。
        - **reduction** (str) - 指定输出结果的计算方式。可选值有：'mean'，'sum'，或'none'。默认值：'mean'。

    输入：
        - **logits** (Tensor) - 输入预测值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。数据类型必须为float16或float32。
        - **labels** (Tensor) - 输入目标值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。与 `logits` 的shape和数据类型相同。

    输出：
        Tensor，数据类型与 `logits` 相同。如果 `reduction` 为'none'，则shape与 `logits` 相同。否则，输出为Scalar的Tensor。

    异常：
        - **TypeError** - `logits` 的数据类型，`labels` 或 `weight` （如果给定）既不是float16，也不是float32。
        - **ValueError** - `reduction` 不为'none'、'mean'或'sum'。
        - **ValueError** - `logits` 的shape与 `labels` 或 `weight` （如果给定）不同。
