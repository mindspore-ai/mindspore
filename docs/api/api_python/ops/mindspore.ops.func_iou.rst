mindspore.ops.iou
=================

.. py:function:: mindspore.ops.iou(anchor_boxes, gt_boxes, mode='iou')

    计算矩形的IOU，即真实区域和预测区域的交并比。

    根据真实区域和预测区域计算IOU(intersection over union)或IOF(intersection over foreground)。

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

        \text{IOF} = \frac{\text{Area of Overlap}}{\text{Area of Ground Truth}}

    .. warning::
        在Ascend中，仅支持计算float16数据。为避免溢出，输入长度和宽度在内部按0.2缩放。

    参数：
        - **anchor_boxes** (Tensor) - 预测区域，shape为(N, 4)的Tensor。"N"表示预测区域的数量，"4"表示"x0"、"y0"、"x1"和"y1"。数据类型为float16、float32或float64。
        - **gt_boxes** (Tensor) - 真实区域，shape为(M, 4)的Tensor。"M"表示地面真实区域的数量，"4"表示"x0"、"y0"、"x1"和"y1"。数据类型为float16、float32或float64。
        - **mode** (string) - 指定计算方法，现支持'iou'(intersection over union)或'iof'(intersection over foreground)模式。默认值：'iou'。
    返回：
        Tensor，IOU/IOF的计算结果，shape为(M, N)的Tensor，数据类型与 `anchor_boxes` 的相同。

    异常：
        - **KeyError** - `mode` 不是'iou'或'iof'。