mindspore.numpy.matmul
======================

.. py:function:: mindspore.numpy.matmul(x1, x2, dtype=None)

    返回两个数组的矩阵乘积。

    .. note::
        不支持NumPy参数 `out` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上，支持的数据类型有np.float16和np.float32。在CPU上，支持的数据类型有np.float16和np.float32。

    参数：
        - **x1** (Tensor) - 输入Tensor，不允许标量。
        - **x2** (Tensor) - 输入Tensor，不允许标量。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，输入的矩阵乘积。 如果 `x1` 和 `x2` 都是1-d向量，返回标量。

    异常：
        - **ValueError** - 如果 `x1` 的最后一个维度的大小不等于 `x2` 的倒数第二个维度的大小，或者传入了标量值。