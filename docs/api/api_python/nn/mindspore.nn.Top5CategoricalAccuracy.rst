mindspore.nn.Top5CategoricalAccuracy
=====================================

.. py:class:: mindspore.nn.Top5CategoricalAccuracy

    计算top-5分类正确率。此类是TopKCategoricalAccuracy的特殊类。有关更多详细信息，请参阅 :class:`TopKCategoricalAccuracy`。

    **样例：**

    >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
    ...            [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
    >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
    >>> topk = nn.Top5CategoricalAccuracy()
    >>> topk.clear()
    >>> topk.update(x, y)
    >>> output = topk.eval()
    >>> print(output)
    1.0
