mindspore.numpy.sqrt
====================

.. py:function:: mindspore.numpy.sqrt(x, dtype=None)

    逐元素返回数组的非负平方根。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在 GPU 上，支持的 dtype 为 np.float16 和 np.float32。

    参数：
        - **x** (Tensor) - 需计算平方根的值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，与 `x` shape相同的数组，包含 `x` 中每个元素的正平方根。对于负元素，返回 nan。如果 `x` 是标量，则返回标量。
