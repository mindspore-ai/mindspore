mindspore.numpy.nanmax
======================

.. py:function:: mindspore.numpy.nanmax(a, axis=None, dtype=None, keepdims=False)

    返回数组的最大值或沿某个轴的最大值，忽略NaN。

    .. note::
        不支持NumPy参数 `out` 。对于全是NaN的slice，返回一个非常小的负数，而不是NaN。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 包含要计算最大值的数的数组。如果 `a` 不是数组，将尝试进行转换。
        - **axis** (Union[int, tuple(int), None], 可选) - 默认值： ``None`` 。计算最大值所沿的单个或多个轴。若取默认值，计算展平数组的最大值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。
        - **keepdims** (boolean, 可选) - 默认值：  ``False`` 。如果设置为 ``True`` ，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和 `a` 同一个维度数。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果 `axis` 超出[-a.ndim, a.ndim)范围，或者 `axis` 包含重复项。