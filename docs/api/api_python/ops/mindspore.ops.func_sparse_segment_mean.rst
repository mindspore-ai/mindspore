mindspore.ops.sparse_segment_mean
=================================

.. py:function:: mindspore.ops.sparse_segment_mean(x, indices, segment_ids)

    `ops.sparse_segment_mean` 从2.8.0版本开始已被弃用，并将在未来版本中被移除。

    计算输入张量稀疏段的平均值。

    .. math::
        output_i = \frac{\sum_j x_{indices[j]}}{N}

    其中 `N` 为满足 :math:`segment\_ids[j] == i` 的元素个数，如果 `segment_ids` 中没有某个值 `i` ，则 :math:`output[i] = 0` 。

    .. note::
        - 在CPU平台， `segment_ids` 必须是升序， `indices` 必须在[0, x.shape[0])范围内。
        - 在GPU平台，对于 `segment_ids` 无需排序，但可能导致未定义的行为。超出范围的 `indices` 将被忽略。

    参数：
        - **x** (Tensor) - 输入tensor，至少为一维。
        - **indices** (Tensor) - 指定索引，一维tensor。
        - **segment_ids** (Tensor) - 一维tensor，必须按升序排列，可包含重复值。

    返回：
        Tensor
