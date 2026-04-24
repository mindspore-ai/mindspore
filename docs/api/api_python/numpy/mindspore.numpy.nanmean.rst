mindspore.numpy.nanmean
=======================

.. py:function:: mindspore.numpy.nanmean(a, axis=None, dtype=None, keepdims=False)

    沿指定轴计算算术平均值，忽略NaN。

    返回数组元素的平均值。默认情况下，平均值在展平的数组上计算，否则在指定轴上计算。对于整数输入，中间值和返回值使用float32类型。

    .. note::
        不支持NumPy参数 `out` 。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 包含要计算均值的数的数组。如果 `a` 不是数组，将尝试进行转换。
        - **axis** (Union[int, tuple of int, None], 可选) - 计算均值所沿的单个或多个轴。若取默认值，计算展平数组的均值。默认值： ``None`` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。
        - **keepdims** (boolean, 可选) - 默认值： ``False`` 。如果设置为 ``True`` ，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和 `a` 同一个维度数。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果 `axis` 超出[-a.ndim, a.ndim)范围，或者 `axis` 包含重复项。