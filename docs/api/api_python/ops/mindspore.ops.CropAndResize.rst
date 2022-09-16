mindspore.ops.CropAndResize
============================

.. py:class:: mindspore.ops.CropAndResize(method="bilinear", extrapolation_value=0.0)

    从输入图像Tensor中提取切片并调整其大小。

    .. note::
        如果输出shape依赖于 `crop_size` 的值，则 `crop_size` 必须为常量。

    参数：
        - **method** (str, 可选) - 指定调整大小的采样方法，为可选字符串。提供的方法有："bilinear"、"nearest"或"bilinear_v2"。"bilinear"代表标准双线性插值算法，而"bilinear_v2"在某些情况下可能会产生更好的结果。默认值："bilinear"。
        - **extrapolation_value** (float, 可选) - 外插值，数据类型为float。默认值：0.0。

    输入：
        - **x** (Tensor) - 输入为四维的Tensor，其shape必须是 :math:`(batch, image\_height, image\_width, depth)` 。支持的数据类型：int8、int16、int32、int64、float16、float32、float64、uint8、uint16。
        - **boxes** (Tensor) - 二维Tensor，其shape为 :math:`(num\_boxes, 4)` 。第i行表示 `box_index[i]` 图像区域的坐标，并且坐标[y1, x1, y2, x2]是归一化后的值。归一化后的坐标值y，映射到图像y * (image_height - 1)处，因此归一化后的图像高度范围为[0, 1]，映射到实际图像高度范围为[0, image_height - 1]。我们允许y1 > y2，在这种情况下，视为原始图像的上下翻转变换。宽度尺寸的处理类似。坐标取值允许在[0, 1]范围之外，在这种情况下，我们使用 `extrapolation_value` 外插值进行补齐。支持的数据类型：float32。
        - **box_index** (Tensor) - `boxes` 的索引，其shape为 :math:`(num\_boxes)` 的一维Tensor，数据类型为int32，取值范围为[0, batch)。box_index[i]的值表示第i个框的图像的值。
        - **crop_size** (Tuple[int]) - 两个int32元素元素组成的tuple：（裁剪高度，裁剪宽度）。只能是常量。所有裁剪后的图像大小都将调整为此大小，且不保留图像内容的宽高比，裁剪高度和裁剪宽度都需要为正。

    输出：
        四维Tensor，其shape为 :math:`(num\_boxes, crop\_height, crop\_width, depth)` ，数据类型类型为float32。

    异常：
        - **TypeError** - 如果 `method` 不是str。
        - **TypeError** - 如果 `extrapolation_value` 不是float，且取值不是"bilinear"、"nearest"或"bilinear_v2"。
        - **ValueError** - 如果 `method` 不是'bilinear'、 'nearest'或者'bilinear_v2'。
