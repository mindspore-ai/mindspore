mindspore.ops.sparse_segment_mean
=================================

.. py:function:: mindspore.ops.sparse_segment_mean(x, indices, segment_ids)

    计算输出Tensor :math:`output_i = \frac{\sum_j x_{indices[j]}}{N}` ，其中平均是对所有 :math:`j` 满足 :math:`segment\_ids[j] == i` 的元素， :math:`N` 表示相加的元素个数。如果给定的分段ID :math:`i` 不存在，则有 :math:`output[i] = 0` 。

    .. note::
        - 在CPU平台， `segment_ids` 中的值会被校验是否排序，若索引值不是升序的，则抛出错误。另外， `indices` 中的值也会被校验是否在界限内，若索引值超出范围[0, x.shape[0])，则抛出错误。
        - 在GPU平台，对于 `segment_ids` 未排序和 `indices` 越界则不抛出错误。如果，无序的 `segment_ids` 会导致安全但未指定的行为，而超出范围的 `indices` 将被忽略。

    参数：
        - **x** (Tensor) - Tensor，其维度必须大于或等于1。
        - **indices** (Tensor) - 一维Tensor，数据类型为int32或int64。
        - **segment_ids** (Tensor) - 一维Tensor，与 `indices` 有同样的数据类型。索引值应当是已排序的，并且可以重复。

    返回：
        Tensor，其数据类型和维数与 `x` 相同。第一维度等于 `segment_ids` 的最后一个元素的值加一，其他维度与 `x` 的对应维度相同。

    异常：
        - **TypeError** - `x` 、 `indices` 或 `segment_ids` 不是Tensor类型。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64的任一类型。
        - **TypeError** - `indices` 和 `segment_ids` 的数据类型不是int32、int64的任一类型。
        - **TypeError** - `indices` 和 `segment_ids` 的数据类型不相同。
        - **ValueError** - `x` 、 `indices` 或 `segment_ids` 的维度不满足上述要求。
        - **ValueError** - `indices` 或 `segment_ids` 的shape不相同。
