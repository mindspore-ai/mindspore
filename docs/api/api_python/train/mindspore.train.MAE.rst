mindspore.train.MAE
====================

.. py:class:: mindspore.train.MAE

    计算平均绝对误差MAE（Mean Absolute Error）。

    计算输入 :math:`x` 和目标 :math:`y` 各元素之间的平均绝对误差。

    .. math::
        \text{MAE} = \frac{\sum_{i=1}^n \|{y\_pred}_i - y_i\|}{n}

    这里， :math:`n` 是batch size。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算平均绝对差（MAE）。

        返回：
            numpy.float64，计算的MAE的结果。

        异常：
            - **RuntimeError** - 样本总数为0。

    .. py:method:: update(*inputs)

        使用预测值 :math:`y_{pred}` 和真实值 :math:`y` 更新局部变量。

        参数：
            - **inputs** - 输入 `y_pred` 和 `y` 来计算MAE，其中 `y_pred` 和 `y` 的shape都是N-D，它们的shape相同。

        异常：
            - **ValueError** - `inputs` 的数量不等于2。
