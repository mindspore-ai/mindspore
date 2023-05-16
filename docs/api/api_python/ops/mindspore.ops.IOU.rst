mindspore.ops.IOU
=================

.. py:class:: mindspore.ops.IOU(mode='iou')

    计算矩形的IOU，即真实区域和预测区域的交并比。

    根据真实区域和预测区域计算IOU(intersection over union)或IOF(intersection over foreground)。

    更多参考详见 :func:`mindspore.ops.iou`。

    参数：
        - **mode** (string) - 指定计算方法，现支持 ``'iou'`` (intersection over union)或 ``'iof'`` (intersection over foreground)模式。默认值： ``'iou'`` 。

    输入：
        - **anchor_boxes** (Tensor) - 预测区域，shape为 :math:`(N, 4)` 的Tensor。"N"表示预测区域的数量，"4"表示"x0"、"y0"、"x1"和"y1"。数据类型为float16或float32。
        - **gt_boxes** (Tensor) - 真实区域，shape为 :math:`(M, 4)` 的Tensor。"M"表示地面真实区域的数量，"4"表示"x0"、"y0"、"x1"和"y1"。数据类型为float16或float32。

    输出：
        IOU值的Tensor，shape为 :math:`(M, N)` 的Tensor，数据类型与 `anchor_boxes` 的相同。
