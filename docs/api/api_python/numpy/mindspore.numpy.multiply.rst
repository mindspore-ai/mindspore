mindspore.numpy.multiply
========================

.. py:function:: mindspore.numpy.multiply(x1, x2, dtype=None)

    参数逐元素相乘。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 需相乘的输入Tensor。
        - **x2** (Tensor) - 需相乘的输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素返回 `x1` 和 `x2` 的乘积。如果 `x1` 和 `x2` 都是标量，返回标量。