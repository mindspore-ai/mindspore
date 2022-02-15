mindspore.nn.Dice
==================

.. py:class:: mindspore.nn.nn.Dice(smooth=1e-05)

    集合相似性度量。

    用于计算两个样本之间的相似性。当分割结果最好时，Dice系数的值为1，当分割结果最差时，Dice系数的值为0。Dice系数表示两个对象之间的面积与总面积的比率。

    .. math::
        dice = \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    **参数：** 

    - **smooth** (float) - 在计算过程中添加到分母里，用于提高数值稳定性，取值需大于0。默认值：1e-5。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算混淆矩阵。

        **返回：**

        Float，计算的结果。

        **异常：**

        - **RuntimeError** - 样本数为0。

    .. py:method:: update(*inputs)

        更新内部评估结果 `y_pred` 和 `y` 。

        **参数：** 

        - **inputs** (tuple) -输入 `y_pred` 和 `y` 。 `y_pred` 和 `y` 是tensor、列表或numpy.ndarray。 `y_pred` 是预测值， `y` 是真实值。

        **异常：**

        - **ValueError** - 输入参数的数量不等于2。
        - **ValueError** - 如果预测值和标签shape不一致。
