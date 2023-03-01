mindspore.ops.NonMaxSuppressionV3
==================================

.. py:class:: mindspore.ops.NonMaxSuppressionV3

    按得分降序排列后采用贪婪策略地选择一组边界框，并剪枝掉与先前选择的框具有高重叠交并比(IOU)的框。
    得分低于 `score_threshold` 的边界框将被删除。

    .. warning::
        如果 `max_output_size` 小于0，其值将置为0。

    .. note::
        - 此算法与原点在坐标系中的位置无关。
        - 对于坐标系的正交变换和平移，该算法不受影响，因此坐标系的平移变换后算法会选择相同的框。

    输入：
        - **boxes** (Tensor) - 二维Tensor，shape为 :math:`(num\_boxes, 4)` 。
        - **scores** (Tensor) - 一个shape为 :math:`(num\_boxes)` 的一维Tensor，表示每个边框（也就是 `boxes` Tensor的每一行）对应的单个分数。 `scores` 中的分数数量必须与 `boxes` 中的边框的数量相等。支持的数据类型为float32。
        - **max_output_size** (Union[Tensor, Number.Int]) - 选取最大的边框数，必须大于等于0，数据类型为int32。
        - **iou_threshold** (Union[Tensor, Number.Float]) - 边框重叠值阈值，重叠值大于此值说明重叠过大。数据类型为float32，值必须在[0, 1]范围内。
        - **score_threshold** (Union[Tensor, Number.Float]) - 移除边框阈值，score值低于此值则边框被移除。支持的数据类型为float32。

    输出：
        一维Tensor，表示被选中边框的index，其shape为 :math:`(M)` ，其中M <= `max_output_size` 。

    异常：
        - **TypeError** -  `boxes` 和 `scores` 的数据类型不一致。
        - **TypeError** - `iou_threshold` 和 `score_threshold` 的数据类型不一致。
        - **TypeError** - `boxes` 的数据类型不是float16或者float32。
        - **TypeError** - `scores` 的数据类型不是float16或者float32。
        - **TypeError** - `max_output_size` 不是Tensor或者Scalar或者其数据类型不是int32或int64。
        - **TypeError** - `iou_threshold` 不是Tesnor或者Scalar，或者其数据类型不是float16或float32。
        - **TypeError** - `score_threshold` 不是Tesnor或者Scalar，或者其数据类型不是float16或float32。
        - **ValueError** - `boxes` 的shape长度不是2或者第二维度的值不是4。
        - **ValueError** - `scores` shape长度不是1。
        - **ValueError** - `max_output_size` 、 `iou_threshold` 或 `score_threshold` 的shape长度不是0。
