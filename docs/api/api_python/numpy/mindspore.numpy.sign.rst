mindspore.numpy.sign
====================

.. py:function:: mindspore.numpy.sign(x, dtype=None)

    逐元素返回数的符号。

    当 `x < 0` 时，sign 函数返回 -1，当 `x == 0` 时，返回 0；当 `x > 0` 时，返回 1。对于 nan 输入，返回 nan。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。目前不支持复数输入。在Ascend上，不支持整数输入。

    参数：
        - **x** (Union[int, float, list, tuple, Tensor]) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        `x` 的符号，当 `x` 是标量时，可以是一个tensor或者标量。

    异常：
        - **TypeError** - 如果输入的 dtype 不在给定类型中或输入不能转换为Tensor。