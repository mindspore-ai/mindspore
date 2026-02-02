mindspore.numpy.trunc
=====================

.. py:function:: mindspore.numpy.trunc(x, dtype=None)

    逐元素返回输入的截断值。

    标量 `x` 的截断值是距离 `x` 最近且趋近于0的整数 `i`，即 `x` 的小数部分被舍弃。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x` 中每个元素的截断值。 如果 `x` 是标量，则返回标量。