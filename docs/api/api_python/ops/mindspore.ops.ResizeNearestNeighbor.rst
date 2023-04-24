mindspore.ops.ResizeNearestNeighbor
=====================================

.. py:class:: mindspore.ops.ResizeNearestNeighbor(size, align_corners=False)

    使用最近邻插值算法调整输入Tensor为指定大小。最近邻插值算法的具体操作为：选择最近点的值，而不考虑相邻点的值，从而产生分段常数插值。

    参数：
        - **size** (Union[tuple, list]) - 指定输入Tensor的新大小。size的维度必须为2。
        - **align_corners** (bool) - 输入和输出Tensor的4个角像素是否居中对齐。默认值： ``False`` 。

    输入：
        - **input_x** (Tensor) - ResizeNearestNeighbor的输入，四维的Tensor，其shape为 :math:`(N, C, H, W)` 。

    输出：
        Tensor，输出Tensor的shape为 :math:`(N, C, NEW\_H, NEW\_W)` 。数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `size` 既不是tuple，也不是list。
        - **TypeError** - `align_corners` 不是bool。
        - **ValueError** - `size` 的shape长度不等于2。