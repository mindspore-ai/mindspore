mindspore.ops.crop_and_resize
=============================

.. py:function:: mindspore.ops.crop_and_resize(image, boxes, box_indices, crop_size, method="bilinear", extrapolation_value=0.0)

    对输入图像Tensor进行裁剪并调整其大小。

    .. note::
        当输出的shape依赖 `crop_size` 的时候，`crop_size` 必须为常量。
        当前该算子的反向仅支持"bilinear"模式，其他模式将会返回0。

    参数：
        - **image** (Tensor) - shape为 :math:`(batch, image\_height, image\_width, depth)` 的图像Tensor。
        - **boxes** (Tensor) - shape为 :math:`(num\_boxes, 4)` 的二维Tensor，表示归一化的边框坐标，坐标格式为 :math:`[y1, x1, y2, x2]` 。其中 :math:`(y1, x1)` 为第一个角点， :math:`(y2, x2)` 为第二个角点。如果 :math:`y1 > y2` ，就是对图像进行的上下翻转，当 :math:`x1 > x2` ，宽度方向操作类似。如果归一化的坐标值超出 :math:`[0, 1]` 的区间，采用 `extrapolation_value` 进行填充。数据类型：float32。
        - **box_indices** (Tensor) - shape为 :math:`(num\_boxes)` 的一维Tensor，表示每个方框的索引。数据类型：int32。
        - **crop_size** (Tuple[int]) - 2元组(crop_height, crop_width)，指定对裁剪出的图像进行调整时的输出大小，元素均为正值。数据类型：int32。
        - **method** (str, 可选) - 指定调整大小的采样方法，为可选字符串。提供的方法有： ``"bilinear"`` 、 ``"nearest"`` 或 ``"bilinear_v2"`` 。默认值： ``"bilinear"`` 。
          
          - ``"nearest"``：最近邻插值。每个输出像素的值为最近的输入像素的值。这种方法简单快速，但可能导致块状或像素化的输出。
          - ``"bilinear"``：双线性插值。每个输出像素是最接近的四个输入像素的加权平均值，使用双线性插值计算。与最近邻插值相比，此方法产生更平滑的结果。
          - ``"bilinear_v2"``：优化后的双线性插值算法，在某些情况下可能会产生更好的结果（更高的精度和速度）。

        - **extrapolation_value** (float，可选) - 指定外插时的浮点值。默认值： ``0.0`` 。

    返回：
        Tensor，shape为 :math:`(num\_boxes, crop\_height, crop\_width, depth)` ，数据类型：float32 。

    异常：
        - **TypeError** - `image`、 `boxes` 或 `box_indices` 不是Tensor。
        - **TypeError** - `crop_size` 不是元素类型为int32的tuple，或 `crop_size` 的长度不为2。
        - **TypeError** - `boxes` 的数据类型不是float，或 `box_indices` 的数据类型不是int32。
        - **TypeError** - `method` 不是字符串。
        - **TypeError** - `extrapolation_value` 不是浮点值。
        - **ValueError** - `image` 的维度不是四维。
        - **ValueError** - `boxes` 的维度不是二维。
        - **ValueError** - `boxes` 的第二维不是4。
        - **ValueError** - `box_indices` 的维度不是一维。
        - **ValueError** - `box_indices` 的第一维与 `boxes` 的第一维不相等。
        - **ValueError** - `box_indices` 存在元素不在 `[0, batch)` 的范围内。
        - **ValueError** - `crop_size` 的数据不是正整数。
        - **ValueError** - `method` 不是 "bilinear"、"nearest"、"bilinear_v2"之一。
