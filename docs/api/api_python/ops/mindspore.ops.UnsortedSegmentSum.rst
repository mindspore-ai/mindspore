mindspore.ops.UnsortedSegmentSum
==================================

.. py:class:: mindspore.ops.UnsortedSegmentSum

    沿分段计算输入Tensor元素的和。

    更多参考详见 :func:`mindspore.ops.unsorted_segment_sum`。

    输入：
        - **input_x** (Tensor) - 待进行分段求和的Tensor，shape： :math:`(x_1, x_2, ..., x_R)` 。
        - **segment_ids** (Tensor) - 用于指示每个元素所属段的标签，将shape设置为 :math:`(x_1, x_2, ..., x_N)` ，其中0<N<=R。
        - **num_segments** (int) - 分段数量 :math:`z` ，可以为int或零维的Tensor。

    输出：
        Tensor，shape为： :math:`(z, x_{N+1}, ..., x_R)` 。
