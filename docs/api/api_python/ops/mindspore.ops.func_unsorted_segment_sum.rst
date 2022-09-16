mindspore.ops.unsorted_segment_sum
====================================

.. py:function:: mindspore.ops.unsorted_segment_sum(input_x, segment_ids, num_segments)

    沿分段计算输入Tensor元素的和。

    计算输出Tensor :math:`\text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]` ，其中 :math:`j,...` 是代表元素索引的Tuple。 `segment_ids` 确定输入Tensor元素的分段。 `segment_ids` 不需要排序，也不需要覆盖 `num_segments` 范围内的所有值。

    UnsortedSegmentSum的计算过程如下图所示：

    .. image:: UnsortedSegmentSum.png

    .. note::
        - 如果 `segment_ids` 中不存在segment_id `i` ，则对输出 `output[i]` 填充0。
        - 在Ascend平台上，如果segment_id的值小于0或大于输入Tensor的shape的长度，将触发执行错误。

    如果给定的segment_ids :math: `i` 的和为空，则：math: `\text{output}[i] = 0` 。如果 `segment_ids` 元素为负数，将忽略该值。 `num_segments` 必须等于不同segment_id的数量。

    参数：
        - **input_x** (Tensor) - shape： :math:`(x_1, x_2, ..., x_R)` 。
        - **segment_ids** (Tensor) - 将形状设置为 :math:`(x_1, x_2, ..., x_N)` ，其中0<N<=R。
        - **num_segments** (Union[int, Tensor], 可选) - 分段数量 :math:`z` ，数据类型为int或0维的Tensor。

    返回：
        Tensor，shape： :math:`(z, x_{N+1}, ..., x_R)` 。

    异常：
        - **TypeError** - `num_segments` 不是int类型。
        - **ValueError** - `segment_ids` 的维度不等于1。

