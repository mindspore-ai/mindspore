mindspore.train.MSE
====================

.. py:class:: mindspore.train.MSE

    测量均方差MSE（Mean Squared Error）。

    计算输入 :math:`y\_pred` 和目标 :math:`y` 各元素之间的平均平方误差。

    .. math::
        \text{MSE}(y\_pred,\ y) = \frac{\sum_{i=1}^n({y\_pred}_i - y_i)^2}{n}

    其中， :math:`n` 为batch size。

    .. py:method:: clear()

        清除内部评估结果。

    .. py:method:: eval()

        计算均方差（MSE）。

        返回：
            numpy.float64，计算的MSE的结果。

        异常：
            - **RuntimeError** - 样本总数为0。

    .. py:method:: update(*inputs)

        使用预测值 :math:`y\_pred` 和真实值 :math:`y` 更新局部变量。

        参数：
            - **inputs** - 输入 :math:`y\_pred` 和 :math:`y` 用于计算MSE。其中 :math:`y\_pred` 和 :math:`y` 的shape都为N-D，它们的shape相同。

        异常：
            - **ValueError** - `inputs` 的数量不等于2。
