mindspore.numpy.ceil
====================

.. py:function:: mindspore.numpy.ceil(x, dtype=None)

    返回输入的向上取整结果，逐元素计算。
    标量 `x` 的向上取整得到最小的整数 `i`，满足 ``i >= x`` 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上，支持的类型有np.float16和np.float32。

    参数：
        - **x** (Tensor) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x` 中每个元素的向上取整结果。如果 `x` 是标量，则结果也是标量。
