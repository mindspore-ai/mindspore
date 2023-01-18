mindspore.ops.ResizeNearestNeighborV2
======================================

.. py:class:: mindspore.ops.ResizeNearestNeighborV2(align_corners=False, half_pixel_centers=False, data_format='NHWC')

    使用最近邻算法将输入Tensor调整为特定大小。

    最近邻算法选择最近点的值，不考虑其他相邻点的值，产生分段常数插值。

    参数：
        - **align_corners** (bool，可选) - 如果为True，则输入输出图像四个角像素的中心被对齐，同时保留角像素处的值。默认值：False。
        - **half_pixel_centers** (bool，可选) - 是否使用半像素中心对齐。如果设置为True，那么 `align_corners` 应该设置为False。默认值：False。
        - **data_format** (str，可选) - 输入 `x` 的数据格式。默认值："NHWC"。

    输入：
        - **x** (Tensor) - 四维的Tensor，其shape为 :math:`(batch, height, width, channels)` 或者 :math:`(batch, channels, height, width)`，取决于 `data_format` 。支持的数据类型列表：[int8, uint8, int16, uint16, int32, int64, float16, float32, float64]。
        - **size** (Tensor) - 输出图片的尺寸。一维的Tensor，含有两个元素 `[new_height, new_width]`。

    输出：
        -  **y** (Tensor) - 调整大小后的图像。是一个shape为 :math:`(batch, new\_height, new\_width, channels)` 或者 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor，具体是哪一个shape取决于 `data_format` 。数据类型与输入 `x` 相同。 

    异常：
        - **TypeError** - `x` 或者 `size` 不是Tensor。
        - **TypeError** - `x` 不在支持的数据类型列表里。
        - **TypeError** - `size` 的数据类型不是int32。
        - **TypeError** - `align_corners` 和 `half_pixel_centers` 不是 bool。
        - **TypeError** - `data_format` 为不是string类型。
        - **ValueError** - `data_format` 不是“NHWC”或者“NCHW”。
        - **ValueError** - `size` 的值含有非正数。
        - **ValueError** - `x` 的维度不等于4。
        - **ValueError** - `size` 的维度不等于1。
        - **ValueError** - `size` 的元素个数不是2。
        - **ValueError** - 属性 `half_pixel_centers` 和 `align_corners` 同时设为True。
