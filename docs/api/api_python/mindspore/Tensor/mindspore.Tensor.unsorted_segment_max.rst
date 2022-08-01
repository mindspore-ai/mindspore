mindspore.Tensor.unsorted_segment_max
=====================================

.. py:method:: mindspore.Tensor.unsorted_segment_max(segment_ids, num_segments)

    沿分段计算输入Tensor的最大值。

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i` ，则将使用 `x` 的数据类型的最小值填充输出 `output[i]` 。
        - `segment_ids` 必须是一个非负Tensor。

    参数：
        - **segment_ids** (Tensor) - shape为 :math:`(x_1)` 的1维张量，值必须是非负数。数据类型支持int32。
        - **num_segments** (int) - 分段的数量。

    返回：
        Tensor，若 `num_segments` 值为 `N` ，则shape为 :math:`(N, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `num_segments` 不是int类型。
        - **ValueError** - `segment_ids` 的维度不等于1。