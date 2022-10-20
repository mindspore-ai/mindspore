mindspore.ops.check_valid
=========================

.. py:function:: mindspore.ops.check_valid(bboxes, img_metas)

    检查边界框。

    检查边界框的交叉数据和数据边界是否有效。

    .. warning::
        指定有效边界 `(长度 * 比率, 宽度 * 比率)` 。

    参数：
        - **bboxes** (Tensor) - shape大小为 :math:`(N, 4)` 。:math:`N` 表示边界框的数量， `4` 表示 `x0` 、`x1` 、`y0` 、`y` 。数据类型必须是float16或float32。
        - **img_metas** (Tensor) - 原始图片的信息 `(长度, 宽度, 比率)` ，需要指定有效边界为 `(长度 * 比率, 宽度 * 比率)` 。数据类型必须是float16或float32。
        
    返回：
        Tensor，shape为 :math:`(N,)` ，类型为bool，需要指出边界框是否在图片内。`True` 表示在，`False` 表示不在。

    异常：
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 不是Tensor。
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 的数据类型既不是float16，也不是float32。
