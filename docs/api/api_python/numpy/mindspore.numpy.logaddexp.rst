mindspore.numpy.logaddexp
=========================

.. py:function:: mindspore.numpy.logaddexp(x1, x2, dtype=None)

    计算输入指数取幂的和的对数。计算 ``log(exp(x1) + exp(x2))`` 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 输入数组。如果 `x1.shape != x2.shape` ，它们必须可以广播到一个共同的shape(即输出的shape)。
        - **dtype** (mindspore.dtype) - 默认值: `None` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。如果 `x1` 和 `x2` 都是标量，返回标量。
