mindspore.ops.bounding_box_encode
=================================

.. py:function:: mindspore.ops.bounding_box_encode(anchor_box, groundtruth_box, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))

    编码边界框位置信息，计算预测边界框和真实边界框之间的偏移，并将此偏移作为损失变量。

    参数：
        - **anchor_box** (Tensor) - 锚框。锚框的shape必须为 :math:`(n, 4)` 。
        - **groundtruth_box** (Tensor) - 真实边界框。它的shape与锚框相同。
        - **means** (tuple，可选) - 计算编码边界框的均值。默认值：(0.0, 0.0, 0.0, 0.0)。
        - **stds** (tuple，可选) - 计算增量的标准偏差。默认值：(1.0, 1.0, 1.0, 1.0)。

    返回：
        Tensor，编码边界框。数据类型和shape与输入 `anchor_box` 相同。

    异常：
        - **TypeError** - 如果 `means` 或 `stds` 不是tuple。
        - **TypeError** - 如果 `anchor_box` 或 `groundtruth_box` 不是Tensor。
