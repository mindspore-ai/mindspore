mindspore.ops.NonMaxSuppressionWithOverlaps
============================================

.. py:class:: mindspore.ops.NonMaxSuppressionWithOverlaps

    贪婪选取一组按score降序排列后的边界框。

    .. note::
        - 此算法与原点在坐标系中的位置无关。
        - 对于坐标系的正交变换和平移，该算法不受影响；因此坐标系的平移变换后算法会选择相同的框。

    输入：
        - **overlaps** (Tensor) - 二维Tensor，其shape为 :math:`(num\_boxes, num_boxes)` ，表示n乘n的边框重叠值。支持的数据类型为float32。
        - **scores** (Tensor) - 一维Tensor，其shape为 :math:`(num\_boxes)` 。表示对应每一行每个方框的score值， `scores` 和 `overlaps` 的num_boxes必须相等。支持的数据类型为float32。
        - **max_output_size** (Union[Tensor, Number.Int]) - 选取最大的边框数，必须大于等于0，数据类型为int32。
        - **overlap_threshold** (Union[Tensor, Number.Float]) - 边框重叠值阈值，重叠值大于此值说明重叠过大。支持的数据类型为float32。
        - **score_threshold** (Union[Tensor, Number.Float]) - 移除边框阈值，边框score值大于此值则移除相应边框。支持的数据类型为float32。

    输出：
        一维Tensor，表示被选中边框的index，其shape为 :math:`(M)` ，其中M <= `max_output_size`。数据类型为int32。

    异常：
        - **TypeError** - `overlaps` 、 `scores` 、 `overlap_threshold` 和 `score_threshold` 的数据类型不是float32。
        - **TypeError** - `overlaps` 或者 `scores` 不是Tesnor。
        - **TypeError** - `max_output_size` 不是Tesnor或者Scalar，或者其数据类型不是int32。
        - **TypeError** - `overlap_threshold` 不是Tesnor或者Scalar，或者其数据类型不是float32。
        - **TypeError** - `score_threshold` 不是Tesnor或者Scalar，或者其数据类型不是float32。
        - **ValueError** - `overlaps` 长度不等于2或者其shape的两个值不相等。
        - **ValueError** - `scores` 的shape长度不是1。
        - **ValueError** - `max_output_size` 、 `overlap_threshold` 或者 `score_threshold` 的shape长度不是1。
        - **ValueError** - `max_output_size` 小于零。
        - **ValueError** - `scores` shape的大小与 `overlaps` 的第零或第一维不相等。
