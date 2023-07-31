mindspore.ops.CropAndResize
===========================

.. py:class:: mindspore.ops.CropAndResize(method="bilinear", extrapolation_value=0.0)

    从输入图像Tensor中提取切片并调整其大小。

    .. note::
        如果输出shape依赖于 `crop_size` 的值，则 `crop_size` 必须为常量。目前该算子仅在 `method` 为"bilinear"支持反向， 其他方法将直接返回0。

    参数：
        - **method** (str, 可选) - 指定调整大小的采样方法，为可选字符串。提供的方法有： ``"bilinear"`` 、 ``"nearest"`` 或 ``"bilinear_v2"`` 。默认值： ``"bilinear"`` 。
          
          - ``"nearest"``：最近邻插值。每个输出像素的值为最近的输入像素的值。这种方法简单快速，但可能导致块状或像素化的输出。
          - ``"bilinear"``：双线性插值。每个输出像素是最接近的四个输入像素的加权平均值，使用双线性插值计算。与最近邻插值相比，此方法产生更平滑的结果。
          - ``"bilinear_v2"``：优化后的双线性插值算法，在某些情况下可能会产生更好的结果（更高的精度和速度）。

        - **extrapolation_value** (float, 可选) - 外插值，数据类型为float。默认值： ``0.0`` 。

    输入：
        - **x** (Tensor) - 输入为四维的Tensor，其shape必须是 :math:`(batch, image\_height, image\_width, depth)` 。支持的数据类型：int8、int16、int32、int64、float16、float32、float64、uint8、uint16。
        - **boxes** (Tensor) - 二维Tensor，其shape为 :math:`(num\_boxes, 4)` 。第i行表示 `box_index[i]` 图像区域的坐标，并且坐标[y1, x1, y2, x2]是归一化后的值。归一化后的坐标值y，映射到图像y * (image_height - 1)处，因此归一化后的图像高度范围为[0, 1]，映射到实际图像高度范围为[0, image_height - 1]。我们允许y1 > y2，在这种情况下，视为原始图像的上下翻转变换。宽度尺寸的处理类似。坐标取值允许在[0, 1]范围之外，在这种情况下，我们使用 `extrapolation_value` 外插值进行补齐。支持的数据类型：float32。
        - **box_index** (Tensor) - `boxes` 的索引，其shape为 :math:`(num\_boxes)` 的一维Tensor，数据类型为int32，取值范围为[0, batch)。box_index[i]的值表示第i个框的图像的值。
        - **crop_size** (Tuple[int]) - 两个int32元素元素组成的tuple：（crop_height, crop_width）。只能是常量。所有裁剪后的图像大小都将调整为此大小，且不保留图像内容的宽高比，裁剪高度和裁剪宽度都需要为正。

    输出：
        四维Tensor，其shape为 :math:`(num\_boxes, crop\_height, crop\_width, depth)` ，数据类型类型为float32。

    异常：
        - **TypeError** - `x`、 `boxes` 或 `box_index` 不是Tensor。
        - **TypeError** - `crop_size` 不是int32类型的tuple，或 `crop_size` 的长度不是2。
        - **TypeError** - `boxes` 的数据类型不是float， 或者，`box_index` 的数据类型不是int32。
        - **TypeError** - `method` 不是字符串。
        - **TypeError** - `extrapolation_value` 不是浮点值。
        - **ValueError** - `x` 的维度不是四维。
        - **ValueError** - `boxes` 的纬度不是二维。
        - **ValueError** - `boxes` 的第二维不是4。
        - **ValueError** - `box_index` 的维度不是一维。
        - **ValueError** - `box_index` 的第一维与 `boxes` 的第一维不相等。
        - **ValueError** - `box_index` 存在元素不在 `[0, batch)` 的范围内.
        - **ValueError** - `crop_size` 的数据不是正整数.
        - **ValueError** - `method` 不是 "bilinear"、"nearest"、"bilinear_v2"之一。
