mindspore.numpy.hypot
=====================

.. py:function:: mindspore.numpy.hypot(x1, x2, dtype=None)

    给定直角三角形的直角边，返回其斜边。

    等价于逐元素计算 ``sqrt(x1**2 + x2**2)`` 。如果 `x1` 或 `x2` 是类标量(即可以无歧义地转换为标量)，它将被广播到另一个参数的每个元素执行计算。(参见示例)

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上，支持的数据类型有np.float16和np.float32。在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **x1** (Tensor) - 三角形的直角边。
        - **x2** (Tensor) - 三角形的直角边。如果 ``x1.shape != x2.shape``，它们必须可以广播到一个共同的shape(这将成为输出的形状)。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。三角形的斜边。如果 `x1` 和 `x2` 都是标量，返回标量。