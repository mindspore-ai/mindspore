mindspore.nn.F1
=====================

.. py:class:: mindspore.nn.F1

    计算F1 score。F1是Fbeta的特殊情况，即beta为1。
    有关更多详细信息，请参阅类:class:`mindspore.nn.Fbeta`。

    .. math::
        F_1=\frac{2\cdot true\_positive}{2\cdot true\_positive + false\_negative + false\_positive}

    **样例：**

    >>> import numpy as np
    >>> from mindspore import nn, Tensor
    >>>
    >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    >>> y = Tensor(np.array([1, 0, 1]))
    >>> metric = nn.F1()
    >>> metric.update(x, y)
    >>> result = metric.eval()
    >>> print(result)
    [0.66666667 0.66666667]
    