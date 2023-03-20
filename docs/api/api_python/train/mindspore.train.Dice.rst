mindspore.train.Dice
=====================

.. py:class:: mindspore.train.Dice(smooth=1e-5)

    集合相似性度量。

    用于计算两个样本之间的相似性。当分割结果最好时，Dice系数的值为1，当分割结果最差时，Dice系数的值为0。Dice系数表示预测值与真实值交集同预测值和真实值并集之间的比值。

    .. math::
        dice = \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    参数：
        - **smooth** (float) - 在计算过程中添加到分母里，用于提高数值稳定性，取值需大于0。默认值：1e-5。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算Dice系数。

        返回：
            Float，计算的结果。

        异常：
            - **RuntimeError** - 样本数为0。

    .. py:method:: update(*inputs)

        更新内部评估结果 `y_pred` 和 `y` 。

        参数：
            - **inputs** (tuple) - 输入 `y_pred` 和 `y` 。 `y_pred` 和 `y` 是tensor、列表或numpy.ndarray。 `y_pred` 是预测值， `y` 是真实值。 `y_pred` 和 `y` 的shape都是 :math:`(N, ...)`。

        异常：
            - **ValueError** - 输入参数的数量不等于2。
            - **ValueError** - 如果预测值和标签shape不一致。
