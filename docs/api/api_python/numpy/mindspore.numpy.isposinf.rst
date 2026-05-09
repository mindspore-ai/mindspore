mindspore.numpy.isposinf
=================================

.. py:function:: mindspore.numpy.isposinf(x)

    逐元素测试是否为正无穷大，并将结果返回为bool数组。

    .. note::
        不支持Numpy的 ``out`` 参数。目前仅支持 ``np.float32`` 。

    参数：
        - **x** (Tensor) - 输入值。

    返回：
        Tensor或标量，当 `x` 为正无穷大时为True，否则为False。如果 `x` 是标量，则返回标量。

    异常：
        - **TypeError** - 如果输入不是Tensor。