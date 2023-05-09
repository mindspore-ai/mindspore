mindspore.ops.CheckValid
=========================

.. py:class:: mindspore.ops.CheckValid

    检查边界框。

    检查由 `bboxes` 指定的一些边框是否是有效的。
    如果边框在由 `img_metas` 确定的边界内部则返回 ``True`` ，否则返回 ``False`` 。


    输入：
        - **bboxes** (Tensor) - shape大小为 :math:`(N, 4)` 。 :math:`N` 表示边界框的数量， `4` 表示 `x0` 、 `y0` 、 `x1` 、 `y1` 。数据类型必须是float16或float32。
        - **img_metas** (Tensor) - 原始图片的信息 :math:`(长度, 宽度, 比率)` ，指定有效边界为 :math:`(长度 * 比率, 宽度 * 比率)` 。数据类型必须是float16或float32。

    输出：
        Tensor，shape为 :math:`(N,)` ，类型为bool，指出边界框是否在图片内。 ``True`` 表示在， ``False`` 表示不在。

    异常：
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 不是Tensor。
        - **TypeError** - 如果 `bboxes` 或者 `img_metas` 的数据类型既不是float16，也不是float32。
