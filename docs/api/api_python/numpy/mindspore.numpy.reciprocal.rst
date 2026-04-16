mindspore.numpy.reciprocal
==========================

.. py:function:: mindspore.numpy.reciprocal(x, dtype=None)

    逐元素返回入参的倒数。

    计算 `1/x` 。

    .. note::
        不支持NumPy参数 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。当使用 `where` 时， `out` 必须具有Tensor值。 `out` 不支持存储结果，但可以与 `where` 结合使用，在 `where` 设置为 `False` 的索引处设定值。

    参数：
        - **x** (Tensor) - 输入数组。对于绝对值大于 1 的整数参数，由于 Python 处理整数除法的方式，结果总是为零。对于整数零，结果溢出。
        - **dtype** (mindspore.dtype, 可选) - 默认值: `None` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，如果 `x` 是标量，则返回标量。