mindspore.ops.ResizeArea
=========================

.. py:class:: mindspore.ops.ResizeArea(align_corners=False)

    使用面积插值调整图像大小到指定的大小。

    调整过程只改变输入图像的高和宽维度数据。

    .. warning::
        `size` 的值必须大于0。

    参数：
        - **align_corners** (bool，可选) - 指定是否对齐输入和输出Tensor的四个角的中心点。当这个参数设置为True时，输出Tensor的角点会和输入Tensor的角点对齐，从而保留角点处的值。默认值：False。

    输入：
        - **images** (Tensor) -输入图像为四维的Tensor，其shape为 :math:`(batch, channels, height, width)` ，数据格式为“NHWC”。支持的数据类型有：int8、int16、int32、int64、float16、float32、float64、uint8和uint16。
        - **size** (Tensor) - 必须为含有两个元素的一维的Tensor，分别为new_height, new_width，表示输出图像的高和宽。支持的数据类型为int32。

    输出：
        Tensor，调整大小后的图像。shape为 :math:`(batch, new\_height, new\_width, channels)` 的四维Tensor，数据类型为float32。 

    异常：
        - **TypeError** - `images` 的数据类型不支持。
        - **TypeError** - `size` 不是int32。
        - **TypeError** - `align_corners` 不是bool。
        - **ValueError** - 输入个数不是2。
        - **ValueError** - `images` 的维度不是4。
        - **ValueError** - `size` 的维度不是1。
        - **ValueError** - `size` 含有元素个数2。
        - **ValueError** - `size` 的元素不全是正数。
