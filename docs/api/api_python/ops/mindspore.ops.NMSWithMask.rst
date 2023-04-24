mindspore.ops.NMSWithMask
===========================

.. py:class:: mindspore.ops.NMSWithMask(iou_threshold=0.5)

    非极大值抑制算法（NMS, Non-maximum Suppression）。当在计算机视觉领域中进行目标检测时，目标检测算法将生成多个边界框，并计算分数最高的边界框与其他边界框的交并比(IOU)，然后根据设定的阈值删除框。
    在Ascend平台上，边界框的分数将被忽略，仅根据框之间的IOU来选择框。这意味着如果要删除分数较低的框，则需要提前按分数对输入框进行降序排序。
    IOU的计算如下：

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

    .. warning::
        一次最多支持2864个输入框。

    参数：
        - **iou_threshold** (float) - 指定删除框的IOU的阈值。默认值： ``0.5`` 。

    输入：
        - **bboxes** (Tensor) - 边界框，shape: :math:`(N, 5)` ， `N` 为边界框的数量。每个边界框包含5个值，前4个值为边界框的坐标（x0、y0、x1、y1），代表左上角和右下角的点。最后一个值为边界框的分数。数据类型支持float16或float32。

    输出：
        tuple[Tensor]，包含三个Tensor：output_boxes、output_idx和selected_mask。

        - **output_boxes** (Tensor) - shape: :math:`(N, 5)` 。在GPU和CPU平台上，它是一个边界框的排序列表，按分数对输入 `bboxes` 进行降序排序。在Ascend平台上，它与输入 `bboxes` 相同。
        - **output_idx** (Tensor) - shape: :math:`(N,)` 。 `output_boxes` 的索引列表。
        - **selected_mask** (Tensor) - shape: :math:`(N,)` 。输出边界框的掩码列表。在 `output_boxes` 上应用此掩码以获取非极大值抑制算法（NMS）计算后的边界框，或在 `output_idx` 上应用此掩码以获取边界框索引。

    异常：
        - **ValueError** - `iou_threshold` 不是float。
        - **ValueError** - 输入Tensor的第一个维度小于或等于0。
        - **TypeError** - `bboxes` 的数据类型非float16或float32。
