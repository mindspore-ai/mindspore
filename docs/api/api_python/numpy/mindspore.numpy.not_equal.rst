mindspore.numpy.not_equal
=================================

.. py:function:: mindspore.numpy.not_equal(x1, x2, dtype=None)

    逐元素返回 ``(x1 != x2)`` 的真值。

    .. note::
        不支持Numpy的 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 和 `extobj` 参数。

    参数：
        - **x1** (Tensor) - 第一个待比较的输入Tensor。
        - **x2** (Tensor) - 第二个待比较的输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，表示 `x1` 和 `x2` 逐元素比较的结果。通常为bool类型，除非传入 `dtype` 。如果 `x1` 和 `x2` 都是标量，则返回标量。

    异常：
        - **TypeError** - 如果输入不是Tensor。