mindspore.ops.multilabel_margin_loss
======================================

.. py:function:: mindspore.ops.multilabel_margin_loss(input, target, reduction='mean')

    用于优化多标签分类问题的合页损失。

    创建一个标准，用于优化输入 :math:`x` （一个2D小批量Tensor）
    和输出 :math:`y` （一个目标类别索引的2DTensor）之间的多标签分类合页损失（基于边距的损失）：
    对于每个小批量样本：

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    其中 :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    并且 :math:`i \neq y[j]` 对于所有 :math:`i` and :math:`j` 。
    :math:`y` 和 :math:`x` shape必须相同。
    该标准仅考虑从前方开始的连续非负目标块。这允许不同的样本具有不同数量的目标类别。

    参数：
        - **input** (Tensor) - 预测值。shape为 :math:`(C)` 或 :math:`(N, C)`，其中 :math:`N`
          为批量大小，:math:`C` 为类别数。数据类型必须为：float16或float32。
        - **target** (Tensor) - 真实标签，shape与 `input` 相同，数据类型必须为int32，标签目标由-1填充。
        - **reduction** (str, 可选) - 可选，对输出应用特定的缩减方法：可选"none"、"mean"、"sum"。默认值：'mean'。

          - 'none'：不应用缩减方法。
          - 'mean'：输出的值总和除以输出的元素个数。
          - 'sum'：输出的总和。

    返回：
        - **outputs** (Union[Tensor, Scalar]) - MultilabelMarginLoss损失。如果 `reduction` 的值为 "none"，
          那么返回shape为 :math:`(N)` 的Tensor类型数据。否则返回一个标量。

    异常：
        - **TypeError** - 当 `input` 或者 `target` 数据不是Tensor时。
        - **TypeError** - 当 `input` 数据类型不是以下其中之一时：float16、float32。
        - **TypeError** - 当 `target` 数据类型不是int32时。
        - **ValueError** - 当 `input` 的数据维度不是以下其中之一时：1、2。
        - **ValueError** - 当 `input` 和 `target` 的shape不相同时。
        - **ValueError** - 当 `reduction` 的值不是以下其中之一时：'none'、 'mean'、 'sum'。
