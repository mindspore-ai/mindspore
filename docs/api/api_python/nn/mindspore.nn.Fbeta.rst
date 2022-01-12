mindspore.nn.Fbeta
==================

.. py:class:: mindspore.nn.Fbeta(beta)

    计算Fbeta评分。

    Fbeta评分是精度(Precision)和召回率(Recall)的加权平均值。

    .. math::
        F_\beta=\frac{(1+\beta^2) \cdot true\_positive}
                {(1+\beta^2) \cdot true\_positive +\beta^2 \cdot false\_negative + false\_positive}

    **参数：**

    - **beta** (Union[float, int]) - F-measure中的beta系数 。

    **样例：**

    >>> import numpy as np
    >>> from mindspore import nn, Tensor
    ...
    >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    >>> y = Tensor(np.array([1, 0, 1]))
    >>> metric = nn.Fbeta(1)
    >>> metric.clear()
    >>> metric.update(x, y)
    >>> fbeta = metric.eval()
    >>> print(fbeta)
    [0.66666667 0.66666667]

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method::eval(average=False)

        计算fbeta结果。

        **参数：**

        - **average** (bool) - 是否计算fbeta平均值。默认值：False。

        **返回：**

        numpy.ndarray或numpy.float64，计算的Fbeta score结果。

    .. py:method:: update(*inputs)

        使用预测值 `y_pred` 和真实标签 `y` 更新内部评估结果。

        **参数：**

        - **inputs** - `y_pred` 和 `y` 。`y_pred` 和 `y` 支持Tensor、list或numpy.ndarray类型。
          通常情况下， `y_pred` 是0到1之间的浮点数列表，shape为 :math:`(N, C)` ，其中 :math:`N` 是样本数， :math:`C` 是类别数。
          `y` 是整数值，如果使用one-hot编码，则shape为 :math:`(N,C)` ；如果使用类别索引，shape是 :math:`(N,)` 。

        **异常：**

          - **ValueError** - 当前输入的 `y_pred` 和历史 `y_pred` 类别数不匹配。
          - **ValueError** - 预测值和真实值包含的类别不同。
