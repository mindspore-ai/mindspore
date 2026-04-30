mindspore.numpy.lcm
===================

.. py:function:: mindspore.numpy.lcm(x1, x2, dtype=None)

    返回 ``|x1|`` 和 ``|x2|`` 的最小公倍数。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入数据。
        - **x2** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None``。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，输入绝对值的最小公倍数。如果 `x1` 和 `x2` 都是标量，返回标量。
