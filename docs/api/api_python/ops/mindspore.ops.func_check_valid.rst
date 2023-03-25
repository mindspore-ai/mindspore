mindspore.ops.check_valid
=========================

.. py:function:: mindspore.ops.check_valid(bboxes, img_metas)

    检查边界框是否在图片内。

    `bboxes` 里包含了多组边界框，每一组边界框用两个横坐标点 :math:`(x0, x1)` 和两个纵坐标点 :math:`(y0, y1)` 表示。
    `img_metas` 提供了原始图片的信息，包含 :math:`(height, width, ratio)` 三个参数，用于指定图片的有效边界。
    当满足：

    :math:`x0 >= 0`

    :math:`y0 >= 0`

    :math:`x1 <= width * ratio - 1`

    :math:`y1 <= height * ratio - 1`

    时，认为检查边界框在图片内。

    .. warning::
        由 `bboxes` 指定的边界框和由 `img_metas` 指定的图片信息需要是有效的，即：
        :math:`x0 <= x1`， :math:`y0 <= y1`， `img_metas` 中的信息 :math:`(height, width, ratio)` 均为正数。

    参数：
        - **bboxes** (Tensor) - shape大小为 :math:`(N, 4)` 。:math:`N` 表示边界框的数量， `4` 表示 :math:`(x0, y0, x1, y1)` 四个坐标点。数据类型必须是float16或float32。
        - **img_metas** (Tensor) - 原始图片的信息 :math:`(height, width, ratio)` ，指定有效边界为 :math:`(height * ratio - 1, width * ratio - 1)` 。数据类型必须是float16或float32。

    返回：
        Tensor，shape为 :math:`(N,)` ，类型为bool，需要指出边界框是否在图片内。`True` 表示在，`False` 表示不在。

    异常：
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 不是Tensor。
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 的数据类型既不是float16，也不是float32。
