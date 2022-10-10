mindspore.ops.crop_and_resize
=============================

.. py:function:: mindspore.ops.crop_and_resize(image, boxes, box_indices, crop_size, method="bilinear", extrapolation_value=0.0)

    对输入图像进行裁剪并调整其大小。

    .. note::
        输出的shape依赖 `crop_size` ，`crop_size` 必须为常量。
        当前该算子的反向仅支持"bilinear"模式，其他模式将会返回0。

    参数：
        - **image** (Tensor) - shape为 :math:`(batch, image_height, image_width, depth)` 的图像Tensor。数据类型：int8, int16, int32, int64, float16, float32, float64, uint8, uint16。
        - **boxes** (Tensor) - shape为 :math:`(num_boxes, 4)` 的2维Tensor。其中，第 :math:`i` 行指定对第 :math:`\text{box_indices[i]}` 张图像裁剪时的归一化坐标 :math:`[y1, x1, y2, x2]`，那么通过归一化的 :math:`y` 坐标值可映射到的图像坐标为 :math:`y * (image\_height - 1)`，因此，归一化的图像高度 :math:`[0, 1]` 间隔映射到的图像高度间隔为 :math:`[0, image\_height - 1]`。我们也允许 :math:`y1 > y2`，这种情况下，就是对图像进行的上下翻转，宽度方向与此类似。同时，我们也允许归一化的坐标值超出 :math:`[0, 1]` 的区间，这种情况下，采用 :math:`\text{extrapolation_value}` 进行填充。数据类型：float32。
        - **box_indices** (Tensor) - shape为 :math:`(num_boxes)` 的1维Tensor，其中，每一个元素必须是 :math:`[0, batch)` 区间内的值。:math:`\test{box_indices[i]}` 指定 :math:`\test{boxes[i, :]}` 所指向的图像索引。数据类型：int32。
        - **crop_size** (Tuple[int]) - 2元组 :math:`(crop_height, crop_width)`，该输入必须为常量并且均为正值。指定对裁剪出的图像进行调整时的输出大小，纵横比可与原图不一致。数据类型：int32。
        - **method** (str) - 指定调整大小时的采样方法，取值为"bilinear"、 "nearest"或"bilinear_v2"，其中，"bilinear"是标准的线性插值算法，而在某些情况下，"bilinear_v2"可能会得到更优的效果。默认值："bilinear"。
        - **extrapolation_value** (float) - 指定外插时的浮点值。默认值： 0.0。

    返回：
        Tensor，shape为 :math:`(num_boxes, crop_height, crop_width, depth)`，数据类型：float32 。

    异常：
        - **TypeError** - `image`、 `boxes` 或 `box_indices` 不是Tensor。
        - **TypeError** - `crop_size` 不是int32的2元组。
        - **TypeError** - `boxes` 的数据类型不是float， 或者，`box_indices` 的数据类型不是int32。
        - **TypeError** - `method` 不是字符串。
        - **TypeError** - `extrapolation_value` 不是浮点值。
        - **ValueError** - `image` 的维度不是4维。
        - **ValueError** - `boxes` 的纬度不是2维。
        - **ValueError** - `boxes` 的第2维不是4。
        - **ValueError** - `box_indices` 的维度不是1维。
        - **ValueError** - `box_indices` 的第1维与 `boxes` 的第1维不相等。
        - **ValueError** - `box_indices` 存在元素不在 `[0, batch)` 的范围内.
        - **ValueError** - `crop_size` 的数据不是正整数.
        - **ValueError** - `method` 不是 "bilinear"、"nearest"、"bilinear_v2"之一。
