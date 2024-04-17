mindspore.ops.ResizeBicubic
============================

.. py:class:: mindspore.ops.ResizeBicubic(align_corners=False, half_pixel_centers=False)

    使用双三次插值调整图像大小到指定的大小。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **align_corners** (bool，可选) - 如果为 ``True`` ，则输入输出图像四个角像素的中心被对齐，同时保留角像素处的值。默认值： ``False`` 。
        - **half_pixel_centers** (bool，可选) - 是否使用半像素中心对齐。如果设置为 ``True`` ，那么 `align_corners` 应该设置为 ``False`` 。默认值： ``False`` 。

    输入：
        - **images** (Tensor) -输入图像为四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，支持的数据类型有：float16、float32、float64。
        - **size** (Union[tuple[int], Tensor[int]]) - tuple或1-D Tensor，含有两个元素，分别为new_height、new_width。推荐使用tuple[int]。

    输出：
        四维Tensor，其shape为 :math:`(batch, channels, new\_height, new\_width)` ，且数据类型与 `images` 一致。

    异常：
        - **TypeError** - `images` 的数据类型不支持。
        - **TypeError** - `align_corners` 不是bool类型。
        - **TypeError** - `half_pixel_centers` 不是bool类型。
        - **ValueError** - `images` 的维度不是4。
        - **ValueError** - 当 `size` 是Tensor时，其维度不是1。
        - **ValueError** - `size` 所含元素的个数不是2。
        - **ValueError** - `size` 中的元素不全是正数。
        - **ValueError** - `align_corners` 和 `half_pixel_centers` 同时为 ``True`` 。
