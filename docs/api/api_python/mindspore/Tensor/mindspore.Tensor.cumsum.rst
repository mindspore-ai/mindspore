mindspore.Tensor.cumsum
=======================

.. py:method:: mindspore.Tensor.cumsum(axis=None, dtype=None)

    返回指定轴方向上元素的累加值。

    .. note::
        如果 `self.dtype` 为 `int8` , `int16` 或 `bool` ，则结果 `dtype` 将提升为 `int32` ，不支持 `int64` 。

    参数：
        - **axis** (int, 可选) - 轴，在该轴方向上的累积和。默认情况下，计算所有元素的累加和。
        - **dtype** (mindspore.dtype, 可选) - 如果未指定参数值，则保持与原始Tensor相同，除非参数值是一个精度小于 `float32` 的整数。在这种情况下，使用 `float32` 。默认值：None。

    返回：
        Tensor。

    异常：
        - **ValueError** - 轴超出范围。