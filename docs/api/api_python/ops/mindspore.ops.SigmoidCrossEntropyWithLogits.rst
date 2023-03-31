mindspore.ops.SigmoidCrossEntropyWithLogits
============================================

.. py:class:: mindspore.ops.SigmoidCrossEntropyWithLogits

    计算预测值与真实值之间的sigmoid交叉熵。

    测量离散分类任务中的分布误差，每个类相互独立，且计算出各个类的交叉熵损失。

    将输入 `logits` 设置为 :math:`X` ，输入 `label` 为 :math:`Y` ，输出为 :math:`loss` 。然后，

    .. math::
        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            loss_{ij} = -[Y_{ij} * ln(p_{ij}) + (1 - Y_{ij})ln(1 - p_{ij})]
        \end{array}

    输入：
        - **logits** (Tensor) - 预测值，任意维度的Tensor，其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。据类型必须为float16或float32。
        - **label** (Tensor) - 真实值。shape和数据类型与 `logits` 的相同。

    输出：
        Tensor，shape和数据类型与输入 `logits` 相同。

    异常：
        - **TypeError** - `logits` 或 `label` 不是Tensor。