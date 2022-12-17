mindspore.Tensor.H
==================

.. py:method:: mindspore.Tensor.H
    :property:

    返回共轭和转置的矩阵（2-D张量）的视图。如果x是复数矩阵，x.H等价于self.swapaxes(0, 1).conj()，如果是实数矩阵则等价于self.swapaxes(0, 1)。
