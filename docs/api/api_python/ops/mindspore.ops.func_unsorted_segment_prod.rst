mindspore.ops.unsorted_segment_prod
===================================

.. py:function:: mindspore.ops.unsorted_segment_prod(x, segment_ids, num_segments)

    分段计算输入tensor的乘积。

    unsorted_segment_prod的计算过程如下图所示：

    .. image:: UnsortedSegmentProd.png

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i` ，则将使用1填充输出 `output[i]` 。
        - `segment_ids` 必须是一个非负tensor。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **segment_ids** (Tensor) - 标识每个元素所属段。
        - **num_segments** (Union[int, Tensor], 可选) - 分段数量，可以为int或零维的tensor。

    返回：
        Tensor
