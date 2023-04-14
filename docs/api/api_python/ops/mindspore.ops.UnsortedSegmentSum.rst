mindspore.ops.UnsortedSegmentSum
==================================

.. py:class:: mindspore.ops.UnsortedSegmentSum

    沿分段计算输入Tensor元素的和。

    更多参考详见 :func:`mindspore.ops.unsorted_segment_sum`。

    输入：
        - **input_x** (Tensor) - shape： :math:`(x_1, x_2, ..., x_R)` 。
        - **segment_ids** (Tensor) - shape为 :math:`(x_1)` 的一维张量，值必须是非负数。数据类型支持int32。
        - **num_segments** (int) - 分段数量 :math:`z` 。

    输出：
        Tensor，shape： :math:`(z, x_{N+1}, ..., x_R)` 。
