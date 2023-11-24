mindspore.ops.UnsortedSegmentMax
================================

.. py:class:: mindspore.ops.UnsortedSegmentMax

    沿分段计算输入Tensor的最大值。

    更多参考详见 :func:`mindspore.ops.unsorted_segment_max`。

    输入：
        - **input_x** (Tensor) - shape： :math:`(x_1, x_2, ..., x_R)` 。
          数据类型支持float16、float32或int32。
        - **segment_ids** (Tensor) - 用于指示每个元素所属段的标签，将shape设置为 :math:`(x_1, x_2, ..., x_N)` ，其中0<N<=R。
          数据类型支持int32。
        - **num_segments** (Union[int, Tensor]) - 分段数量 :math:`z` ，可以为int或零维的Tensor。

    输出：
        Tensor，shape为： :math:`(z, x_{N+1}, ..., x_R)` 。
