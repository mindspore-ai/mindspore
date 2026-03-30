mindspore.ops.unsorted_segment_max
==================================

.. py:function:: mindspore.ops.unsorted_segment_max(x, segment_ids, num_segments)

    分段计算输入tensor的最大值。

    unsorted_segment_max的计算过程如下图所示：

    .. image:: UnsortedSegmentMax.png

    .. math::
        \text { output }_i=\text{max}_{j \ldots} \text { data }[j \ldots]

    :math:`max` 返回元素 :math:`j...` 中的最大值，其中 :math:`segment\_ids[j...] == i` 。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i`，则输出 `output[i]` 将使用 `x` 的数据类型的最小值填充。
        - `segment_ids` 必须是一个非负tensor。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **segment_ids** (Tensor) - 标识每个元素所属段。
        - **num_segments** (Union[int, Tensor]) - 分段数量，可以为int或零维的tensor。

    返回：
        Tensor
