mindspore.numpy.nansum
======================

.. py:function:: mindspore.numpy.nansum(a, axis=None, dtype=None, keepdims=False)

    计算指定轴上元素的总和，将NaN(非数值)视为零。

    .. note::
        不支持NumPy参数 `out` 。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 包含要计算总和的数的数组。如果 `a` 不是数组，将尝试进行转换。
        - **axis** (Union[int, tuple of int, None], 可选) - 计算总和所沿的单个或多个轴。若取默认值，计算展平数组的总和。默认值: ``None`` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。 覆盖输出Tensor的dtype。
        - **keepdims** (boolean, 可选) - 默认值:  ``False`` 。如果设置为 ``True`` ，减少的轴在结果中保留为大小为1的维度。 若使用此选项，结果会广播到和 `a` 同一个维度数。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果 `axis` 超出[-a.ndim, a.ndim)范围，或者 `axis` 包含重复项。