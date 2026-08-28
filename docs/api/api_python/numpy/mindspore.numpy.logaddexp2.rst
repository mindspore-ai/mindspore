mindspore.numpy.logaddexp2
==========================

.. py:function:: mindspore.numpy.logaddexp2(x1, x2, dtype=None)

    计算输入指数取幂的和的以2为底对数。

    计算 ``log2(2**x1 + 2**x2)`` 。在机器学习中，这个函数在计算的概率非常小以至于可能超出正常浮点数范围时很有用。即当概率值很小时，使用概率值的以2为底的对数表示。这个函数以这种方式存储概率值和。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入Tensor。
        - **x2** (Tensor) - 输入Tensor。如果 ``x1.shape != x2.shape`` ，则它们必须可以广播到一个共同的shape（即输出的shape）。
        - **dtype** (mindspore.dtype) - 默认值: `None` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。如果 `x1` 和 `x2` 都是标量，返回标量。