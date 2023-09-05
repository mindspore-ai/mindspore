mindspore.nn.SoftMarginLoss
============================

.. py:class:: mindspore.nn.SoftMarginLoss(reduction='mean')

    针对二分类问题的损失函数。

    SoftMarginLoss用于计算输入Tensor :math:`x` 和目标值Tensor :math:`y` （包含1或-1）的二分类损失值。

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{x.nelement()}

    :math:`x.nelement()` 代表 `x` 中元素的个数。

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 预测值，数据类型为float16或float32。
        - **labels** (Tensor) - 目标值，数据类型和shape与 `logits` 的相同。

    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape与 `logits` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - `logits` 或 `labels` 不是Tensor。
        - **TypeError** - `logits` 或 `labels` 的数据类型既不是float16也不是float32。
        - **ValueError** - `logits` 的shape与 `labels` 不同。
        - **ValueError** - `reduction` 不为"mean"，"sum"，或"none"。