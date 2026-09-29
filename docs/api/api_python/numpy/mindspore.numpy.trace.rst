mindspore.numpy.trace
=================================

.. py:function:: mindspore.numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None)

    返回张量沿对角线的元素之和。
    如果 ``a`` 是二维数组，则返回沿给定偏移量的对角线的元素之和，即所有 ``i`` 对应的元素 ``a[i,i+offset]`` 的和。如果 ``a`` 有超过两个维度，则使用 ``axis1`` 和 ``axis2`` 指定的轴来确定返回的二维子数组的迹。结果数组的shape与移除 ``axis1`` 和 ``axis2`` 后的 ``a`` 相同。

    .. note::
        当前仅用于 MindScience 科学计算场景，不支持其他使用场景。
        暂不支持 Windows 平台。在GPU上，支持的数据类型为 ``np.float16`` 和 ``np.float32``。在CPU上，支持的数据类型为 ``np.float16`` 、 ``np.float32`` 和 ``np.float64`` 。

    参数：
        - **a** (Tensor) - 输入的需要计算的矩阵。
        - **offset** (int, 可选) - 对角线相对于主对角线的偏移量。可以为正或负。默认是主对角线。
        - **axis1** (int, 可选) - 用于二维子数组的第一个轴，从中提取对角线。默认是第一个轴(0)。
        - **axis2** (int, 可选) - 用于二维子数组的第二个轴，从中提取对角线。默认是第二个轴。
        - **dtype** (mindspore.dtype, 可选) - 指定Tensor的数据类型，若不为 ``None`` 则重写Tensor的数据类型。默认值： ``None`` 。

    返回：
        Tensor，沿对角线的元素之和。如果传递的数组 ``a`` 是二维数组，则返回主对角线元素的总和。如果 ``a`` 有更高的维度，则返回沿对角线的和组成的数组。

    异常：
        - **ValueError** - 如果输入Tensor的维度小于二维。
        - **ValueError** - 如果 ``axis1`` 或 ``axis2`` 不在 [-dims, dims)范围内，其中dims的维度为 `a`。
        - **ValueError** - 如果 ``axis1`` 和 ``axis2`` 指定的轴相同。
