mindspore.numpy.mean
====================

.. py:function:: mindspore.numpy.mean(a, axis=None, keepdims=False, dtype=None)

    沿指定轴计算算术平均值。

    返回数组元素的平均值。默认情况下，平均值是在展平的数组上计算的，否则在指定的轴上计算。

    .. note::
        不支持NumPy参数 `out` 。在GPU上，支持的数据类型有np.float16和np.float32。

    参数：
        - **a** (Tensor) - 包含要计算均值的数字的输入Tensor。如果 `a` 不是数组，将尝试进行转换。
        - **axis** (Union[int, tuple(int), None], 可选) - 计算均值的所沿的一个或多个轴。默认计算展平数组的均值。如果这是一个整数tuple，将在多个轴上计算均值。默认值： ``None`` 。
        - **keepdims** (bool, 可选) - 如果设置为 ``True``，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和输入Tensor同一个维度数。默认值： ``False`` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，包含元素为所得均值的数组。

    异常：
        - **ValueError** - 如果 `axis` 的范围超过 `[-a.ndim, a.ndim)` ，或如果 `axis` 包含重复项。