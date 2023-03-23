mindspore.Tensor.H
==================

.. py:method:: mindspore.Tensor.H
    :property:

    返回共轭和转置的矩阵（2-D Tensor）的视图。如果输入x是复数矩阵，x.H等价于 `mindspore.Tensor.swapaxes(0, 1).conj()`，如果是实数矩阵则等价于 `mindspore.Tensor.swapaxes(0, 1)`。
