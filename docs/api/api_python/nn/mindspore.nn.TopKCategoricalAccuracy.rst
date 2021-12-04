mindspore.nn.TopKCategoricalAccuracy
====================================

.. py:class:: mindspore.nn.TopKCategoricalAccuracy(k)

    计算top-k分类正确率。

    .. note::
        `update` 方法需要接收满足 :math:`(y_{pred}, y)` 格式的输入。如果某些样本具有相同的正确率，则将选择第一个样本。

    **参数：**

    - **k (int)** - 指定要计算的top-k分类正确率。

    **异常：**

    - **TypeError** - `k` 不是int。
    - **ValueError** - `k` 小于1。

    **样例：**

    >>> import numpy as np
    >>> from mindspore import nn, Tensor
    >>>
    >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
    ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
    >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
    >>> topk = nn.TopKCategoricalAccuracy(3)
    >>> topk.clear()
    >>> topk.update(x, y)
    >>> output = topk.eval()
    >>> print(output)
    0.6666666666666666

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算top-k分类正确率。

        **返回：**

        numpy.float64，计算结果。

    .. py:method:: update(*inputs)

        使用预测值 `y_pred` 和真实标签 `y` 更新局部变量。

        **参数：**

        - **inputs** - 输入 `y_pred` 和 `y`。`y_pred` 和 `y` 支持Tensor、list或numpy.ndarray类型。
          `y_pred` 在大多数情况下由范围 :math:`[0, 1]` 中的浮点数组成，shape为 :math:`(N, C)` ，其中 :math:`N` 是样本数， :math:`C` 是类别数。
          `y` 由整数值组成。如果使用one-hot编码，则shape为 :math:`(N, C)` ；如果使用类别索引，shape是 :math:`(N,)` 。
