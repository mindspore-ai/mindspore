mindspore.nn.ResizeBilinear
============================

.. py:class:: mindspore.nn.ResizeBilinear(half_pixel_centers=False)

    使用双线性插值调整输入Tensor为指定的大小。

    参数：
        - **half_pixel_centers** (bool) - 是否几何中心对齐。如果设置为True, 那么 `align_corners` 应该设置为False。默认值：False。

    输入：
        - **x** (Tensor) - ResizeBilinear的输入，四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，数据类型为float16或float32。
        - **size** (Union[tuple[int], list[int], None]) - 指定新Tensor的shape大小，其shape为 :math:`(new\_height, new\_width)` 的tuple或者list。只有size或scale_factor能设置为None。默认值：None。
        - **scale_factor** (int, None) - 新Tensor大小的缩放因子，其值为正整数。 `size` 或 `scale_factor` 有且只有一个能设置为None。默认值：None。
        - **align_corners** (bool) - 如果为True，将使用 :math:`(new\_height - 1) / (height - 1)` 来调整输入，这将精确对齐图像的4个角以及调整图像尺寸。如果为False，将使用 :math:`new\_height / height` 来调整。默认值：False。

    输出：
        调整后的Tensor。

        如果设置了size，则结果为 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor，其数据类型与 `x` 相同。如果设置了scale，则结果为 :math:`(batch, channels, scale\_factor * height, scale\_factor * width)` 的四维Tensor，其数据类型与 `x` 相同。

    异常：
        - **TypeError** - `size` 不是tuple、list或None。
        - **TypeError** - `scale_factor` 既不是int也不是None。
        - **TypeError** - `align_corners` 不是bool。
        - **TypeError** - `half_pixel_centers` 不是bool。
        - **TypeError** - `align_corners` 和 `half_pixel_centers` 都为True。
        - **TypeError** - `half_pixel_centers` 为True，且device_target不为Ascend。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - `size` 和 `scale_factor` 都为None或都不为None。
        - **ValueError** - `x` 的shape长度不等于4。
        - **ValueError** - `scale_factor` 是小于0的int。
        - **ValueError** - `size` 是长度不等于2的list或tuple。
