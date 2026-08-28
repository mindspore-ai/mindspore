mindspore.numpy.greater_equal
=================================

.. py:function:: mindspore.numpy.greater_equal(x1, x2, dtype=None)

    逐元素返回 :math:`(x1 >= x2)` 的真值。

    .. note::
        不支持Numpy的 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 和 `extobj` 参数。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 输入数组。如果 :math:`x1.shape != x2.shape` ，则它们必须能够广播到一个共同的shape（该shape成为输出的shape）。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x1` 和 `x2` 逐元素比较的结果。通常为bool类型，除非传递了 `dtype` 。如果 `x1` 和 `x2` 都是标量，则返回标量。