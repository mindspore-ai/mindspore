mindspore.ops.ResizeNearestNeighborV2
======================================

.. py:class:: mindspore.ops.ResizeNearestNeighborV2(align_corners=False, half_pixel_centers=False)

    使用最近邻算法将输入Tensor调整为特定大小。

    最近邻算法选择最近点的值，不考虑其他相邻点的值，产生分段常数插值。

    参数：
        - **align_corners** (bool，可选) - 如果为 ``True`` ，则输入输出图像四个角像素的中心被对齐，同时保留角像素处的值。默认值： ``False`` 。
        - **half_pixel_centers** (bool，可选) - 是否使用半像素中心对齐。如果设置为 ``True`` ，那么 `align_corners` 应该设置为 ``False`` 。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 四维的Tensor，其shape为 :math:`(batch, channels, height, width)`。
        - **size** (Tensor) - 输出图片的尺寸。一维的Tensor，含有两个元素[ `new_height` , `new_width` ]。

    输出：
        -  **y** (Tensor) - 调整大小后的图像。是一个shape为 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor。数据类型与输入 `x` 相同。 

    异常：
        - **TypeError** - `x` 或者 `size` 不是Tensor。
        - **TypeError** - `size` 的数据类型不是int32。
        - **TypeError** - `align_corners` 和 `half_pixel_centers` 不是 bool。
        - **ValueError** - `size` 的值含有非正数。
        - **ValueError** - `x` 的维度不等于4。
        - **ValueError** - `size` 的维度不等于1。
        - **ValueError** - `size` 的元素个数不是2。
        - **ValueError** - 属性 `half_pixel_centers` 和 `align_corners` 同时设为True。
