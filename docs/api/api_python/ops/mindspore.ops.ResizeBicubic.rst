mindspore.ops.ResizeBicubic
============================

.. py:class:: mindspore.ops.ResizeBicubic(align_corners=False, half_pixel_centers=False)

    使用双三次插值调整图像大小到指定的大小。

    .. warning::
        输出最大长度为1000000。

    参数：
        - **align_corners** (bool，可选) - 如果为True，则输入输出图像四个角像素的中心被对齐，同时保留角像素处的值。默认值：False。
        - **half_pixel_centers** (bool，可选) - 是否使用半像素中心对齐。如果设置为True，那么 `align_corners` 应该设置为False。默认值：False。

    输入：
        - **images** (Tensor) -输入图像为四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，支持的数据类型有：int8、int16、int32、int64、float16、float32、float64、uint8和uint16。
        - **size** (Tensor) - 必须为含有两个元素的一维的Tensor，分别为new_height, new_width，表示输出图像的高和宽。支持的数据类型为int32。

    输出：
        Tensor，调整大小后的图像。shape为 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor，数据类型为float32。 

    异常：
        - **TypeError** - `images` 的数据类型不支持。
        - **TypeError** - `size` 不是int32。
        - **TypeError** - `align_corners` 不是bool。
        - **TypeError** - `half_pixel_centers` 不是bool。
        - **ValueError** - `images` 的维度不是4。
        - **ValueError** - `size` 的维度不是1。
        - **ValueError** - `size` 含有元素个数数不是2。
        - **ValueError** - `size` 的元素不全是正数。
        - **ValueError** - `align_corners` 和 `half_pixel_centers` 同时为True。
