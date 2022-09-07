mindspore.ops.SoftMarginLoss
=============================

.. py:class:: mindspore.ops.SoftMarginLoss(reduction='mean')

    SoftMarginLoss操作。

    一个二分类任务，计算输入 :math:`x` 和真实标签 :math:`y` （包含1或-1）之间的损失。

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    其中 :math:`x.nelement()` 是x的元素数量。

    参数：
        - **reduction** (str) - 指定输出结果的计算方式。可选值有：'none'、'mean'或'sum'。默认值：'mean'。

    输入：
        - **logits** (Tensor) - 预测值。数据类型必须为float16或float32。
        - **labels** (Tensor) - 真实标签，数据类型和shape与 `logits` 相同。

    输出：
        Tensor或Scalar，如果 `reduction` 为'none'，其shape与 `logits` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - 如果 `logits` 或 `labels` 不是Tensor。
        - **TypeError** - 如果 `logits` 或 `labels` 的数据类型既不是float16也不是float32。
        - **ValueError** - 如果 `logits` 与 `labels` 的shape不相同。
        - **ValueError** - 如果 `reduction` 不是'none'、'mean'或'sum'。
