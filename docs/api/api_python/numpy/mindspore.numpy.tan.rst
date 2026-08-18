mindspore.numpy.tan
===================

.. py:function:: mindspore.numpy.tan(x, dtype=None)

    逐元素计算正切值。

    等价于 :math:`np.\sin(x)/np.\cos(x)` ，逐元素计算。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。如果 `x` 是标量，则返回标量。

    异常：
        - **TypeError** - 如果输入不是Tensor或Tensor的 dtype 为 mindspore.float64。