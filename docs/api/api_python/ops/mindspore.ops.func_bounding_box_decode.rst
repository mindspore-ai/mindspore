mindspore.ops.bounding_box_decode
=================================

.. py:function:: mindspore.ops.bounding_box_decode(anchor_box, deltas, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016)

    解码边界框位置信息，计算偏移量，此算子将偏移量转换为Bbox，用于在后续图像中标记目标等。

    参数：
        - **anchor_box** (Tensor) - 锚框。锚框的shape必须为 :math:`(n, 4)` 。
        - **deltas** (Tensor) - 框的增量。它的shape与 `anchor_box` 相同。
        - **max_shape** (tuple) - 解码框计算的上限值。
        - **means** (tuple，可选) - 计算 `deltas` 的均值。默认值：(0.0, 0.0, 0.0, 0.0)。
        - **stds** (tuple，可选) - 计算 `deltas` 的标准差。默认值：(1.0, 1.0, 1.0, 1.0)。
        - **wh_ratio_clip** (float，可选) - 解码框计算的宽高比限制。默认值：0.016。

    返回：
        Tensor，解码框。它的数据类型和shape与 `anchor_box` 相同。

    异常：
        - **TypeError** - 如果 `means` 、 `stds` 或 `max_shape` 不是tuple。
        - **TypeError** - 如果 `wh_ratio_clip` 不是float。
        - **TypeError** - 如果 `anchor_box` 或 `deltas` 不是Tensor。
