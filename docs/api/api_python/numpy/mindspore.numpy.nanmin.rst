mindspore.numpy.nanmin
======================

.. py:function:: mindspore.numpy.nanmin(a, axis=None, dtype=None, keepdims=False)

    返回数组的最小值或沿某个轴的最小值，忽略NaN。

    .. note::
        不支持NumPy参数 `out` 。对于全是NaN的slice，返回一个非常小的负数，而不是NaN。在Ascend上，由于目前不支持检查NaN，不推荐使用np.nanmin。如果数组不包含NaN，应使用np.min。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 包含要计算最小值的数的数组。如果 `a` 不是数组，将尝试进行转换。
        - **axis** (Union[int, tuple(int), None], 可选) - 计算最小值所沿的单个或多个轴。若取默认值，计算展平数组的最小值。默认值： ``None`` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。
        - **keepdims** (boolean, 可选) - 默认值： ``False`` 。如果设置为 `True` ，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和 `a` 同一个维度数。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果 `axis` 超出[-a.ndim, a.ndim)范围，或者 `axis` 包含重复项。