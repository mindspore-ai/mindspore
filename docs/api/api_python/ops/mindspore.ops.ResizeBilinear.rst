mindspore.ops.ResizeBilinear
=============================

.. py:class:: mindspore.ops.ResizeBilinear(size, align_corners=False, half_pixel_centers=False)

    使用双线性插值调整图像大小到指定的大小。

    调整输入图像的高和宽。且可以输入不同数据类型的图像，但输出的数据类型只能是float32。

    使用通用resize功能请参考 :func:`mindspore.ops.interpolate`。

    .. warning::
        这个接口不支持动态shape，而且将来可能面临更改或删除，用 :func:`mindspore.ops.interpolate` 可替代该接口。

    参数：
        - **size** (Union[tuple[int], list[int]]) - 指定图像的新尺寸，输入格式为：2个int元素 :math:`(new\_height, new\_width)` 的tuple或者list。
        - **align_corners** (bool) - 如果为True，则通过 :math:`(new\_height - 1) / (height - 1)` 调整输入，这将精确对齐图像的4个角和调整图像大小。如果为False，则按 :math:`new\_height / height` 调整输入。默认值：False。
        - **half_pixel_centers** (bool) - 是否几何中心对齐。如果设置为True, 那么 `scale_factor` 应该设置为False。默认值：False。

    输入：
        - **x** (Tensor) - ResizeBilinear的输入，四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，数据类型为float32或float16。

    输出：
        Tensor，调整大小后的图像。shape为 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor，数据类型与输入 `x` 相同。 

    异常：
        - **TypeError** - `size` 既不是tuple，也不是list。
        - **TypeError** - `align_corners` 不是bool。
        - **TypeError** - `half_pixel_centers` 不是bool。
        - **TypeError** - `align_corners` 和 `half_pixel_centers` 都为True。
        - **TypeError** - `half_pixel_centers` 为True，且device_target不为Ascend。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 的shape长度不等于4。