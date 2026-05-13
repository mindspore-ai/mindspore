mindspore.numpy.isnan
=================================

.. py:function:: mindspore.numpy.isnan(x, dtype=None)

    逐元素测试是否为NaN，并将结果返回为bool数组。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。目前仅支持 ``np.float32`` 。

    参数：
        - **x** (Tensor) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，当 `x` 为NaN时为True，否则为False。如果 `x` 是标量，则返回标量。