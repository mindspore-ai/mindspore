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
        - **segment_ids** (Tensor) - shape为 :math:`(x_1)` 的一维张量，值必须是非负数。数据类型支持int32。
        - **num_segments** (int) - 分段的数量。必须大于0。

    返回：
        Tensor，若 `num_segments` 值为 `N` ，则shape为 :math:`(N, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `num_segments` 不是int类型。
        - **ValueError** - `segment_ids` 的维度不等于1。
