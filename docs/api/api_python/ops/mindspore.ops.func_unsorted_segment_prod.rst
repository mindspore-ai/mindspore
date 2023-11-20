mindspore.ops.unsorted_segment_prod
===================================

.. py:function:: mindspore.ops.unsorted_segment_prod(x, segment_ids, num_segments)

    沿分段计算输入Tensor元素的乘积。

    unsorted_segment_prod的计算过程如下图所示：

    .. image:: UnsortedSegmentProd.png

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i` ，则将使用1填充输出 `output[i]` 。
        - `segment_ids` 必须是一个非负Tensor。

    参数：
        - **x** (Tensor) - shape： :math:`(x_1, x_2, ..., x_R)` 。数据类型支持float16、float32或int32。
        - **segment_ids** (Tensor) - 用于指示每个元素所属段的标签，将shape设置为 :math:`(x_1, x_2, ..., x_N)` ，其中0<N<=R。数据类型支持int32。
        - **num_segments** (Union[int, Tensor], 可选) - 分段数量 :math:`z` ，可以为int或零维的Tensor。

    返回：
        Tensor, shape为 :math:`(z, x_{N+1}, ..., x_R)`。

    异常：
        - **TypeError** - `num_segments` 不是int类型。
