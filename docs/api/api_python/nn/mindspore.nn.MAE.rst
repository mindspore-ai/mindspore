mindspore.nn.MAE
================

.. py:class:: mindspore.nn.MAE

    计算平均绝对误差MAE（Mean Absolute Error）。

    计算输入 :math:`x` 和目标 :math:`y` 各元素之间的平均绝对误差。

    .. math::
        \text{MAE} = \frac{\sum_{i=1}^n \|y_{pred}_i - y_i\|}{n}

    这里， :math:`n` 是batch size。

    **样例：**

    >>> import numpy as np
    >>> from mindspore import nn, Tensor
    >>>
    >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
    >>> y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]), mindspore.float32)
    >>> error = nn.MAE()
    >>> error.clear()
    >>> error.update(x, y)
    >>> result = error.eval()
    >>> print(result)
    0.037499990314245224

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算平均绝对差（MAE）。

        **返回：**

        numpy.float64，计算的MAE的结果。

        **异常：**

        - **RuntimeError** - 样本总数为0。

    .. py:method:: update(*inputs)

        使用预测值 :math:`y_{pred}` 和真实值 :math:`y` 更新局部变量。

        **参数：**

        - **inputs** - 输入 `y_pred` 和 `y` 来计算MAE，其中 `y_pred` 和 `y` 的shape都是N-D，它们的shape相同。

        **异常：**

        - **ValueError** - `inputs` 的数量不等于2。
