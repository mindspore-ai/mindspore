mindspore.ops.ResizeBilinearV2
===============================

.. py:class:: mindspore.ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=False)

    使用双线性插值调整图像大小到指定的大小。

    调整过程只改变输入图像最低量维度的数据，分别代表高和宽。

    .. note::
        目前不支持动态shape特性。

    参数：
        - **align_corners** (bool，可选) - 如果为True，则使用比例 :math:`(new\_height - 1) / (height - 1)` 对输入进行缩放，此时输入图像和输出图像的四个角严格对齐。如果为False，使用比例 :math:`new\_height / height` 输入进行缩放。默认值：False。
        - **half_pixel_centers** (bool，可选) - 是否使用半像素中心对齐。如果设置为True，那么 `align_corners` 应该设置为False。默认值：False。

    输入：
        - **x** (Tensor) -输入图像为四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，支持的数据类型有：float16、float32。
        - **size** (Union[tuple[int], list[int], Tensor]) - 调整后图像的尺寸。为含有两个元素的一维的Tensor或者list或者tuple，分别为 :math:`(new\_height, new\_width)` 。

    输出：
        Tensor，调整大小后的图像。shape为 :math:`(batch, channels, new\_height, new\_width)` 的四维Tensor，数据类型与 `x` 一致。 

    异常：
        - **TypeError** - `align_corners` 不是bool。
        - **TypeError** - `half_pixel_centers` 不是bool。
        - **TypeError** - `align_corners` 和 `half_pixel_centers` 同时为True。
        - **ValueError** - `half_pixel_centers` 为True，同时运行平台为CPU。
        - **ValueError** - `x` 维度不是4。
        - **ValueError** - `size` 为Tensor且维度不是1。
        - **ValueError** - `size` 含有元素个数不是2。
