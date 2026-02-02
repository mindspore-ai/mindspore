mindspore.numpy.unravel_index
=================================

.. py:function:: mindspore.numpy.unravel_index(indices, shape, order='C')

    将一维索引或一维索引数组转换为坐标数组的tuple。

    .. note::
        超出边界的索引会被裁剪到 `shape` 的边界，而不是引发错误。

    参数：
        - **indices** (Union[int, float, bool, list, tuple, Tensor]) - 一个整数数组，其元素是展平后的数组中对应 `shape` 维度位置的索引。
        - **shape** (tuple(int)) - 用于将索引转换为坐标的数组shape。
        - **order** (Union['C', 'F'], 可选) - 确定索引是否按行主序（C-style）或列主序（Fortran-style）进行处理。默认值： ``'C'`` 。

    返回：
        Tensor，包含坐标数组的tuple，每个数组的shape与索引数组相同。

    异常：
        - **ValueError** - 如果 `order` 不是 ``'C'`` 或 ``'F'`` 。