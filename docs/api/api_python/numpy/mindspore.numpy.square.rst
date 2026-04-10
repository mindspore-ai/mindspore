mindspore.numpy.square
======================

.. py:function:: mindspore.numpy.square(x, dtype=None)

    逐元素返回输入的平方。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上，支持的dtype为np.float16和np.float32。

    参数：
        - **x** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素计算 ``x*x`` ，具有与 `x` 相同的shape和dtype。如果 `x` 是标量，则返回标量。