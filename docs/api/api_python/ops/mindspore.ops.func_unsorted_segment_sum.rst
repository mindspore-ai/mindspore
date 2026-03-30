mindspore.ops.unsorted_segment_sum
====================================

.. py:function:: mindspore.ops.unsorted_segment_sum(input_x, segment_ids, num_segments)

    分段计算输入tensor的和。

    unsorted_segment_sum的计算过程如下图所示：

    .. image:: UnsortedSegmentSum.png

    .. math::
        \text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]

    其中 :math:`j,...` 是代表元素索引的tuple。 `segment_ids` 确定输入tensor元素的分段。 `segment_ids` 不需要排序，也不需要覆盖 `num_segments` 范围内的所有值。

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i` ，则对输出 `output[i]` 填充0。
        - 在Ascend平台上，如果segment_id的值小于0或大于输入tensor的shape的长度，将触发执行错误。

    如果给定的segment_ids :math:`i` 的和为空，则 :math:`\text{output}[i] = 0` 。如果 `segment_ids` 元素为负数，将忽略该值。 `num_segments` 必须等于不同segment_id的数量。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **segment_ids** (Tensor) - 标识每个元素所属段。
        - **num_segments** (Union[int, Tensor], 可选) - 分段数量，可以为int或零维的tensor。

    返回：
        Tensor
