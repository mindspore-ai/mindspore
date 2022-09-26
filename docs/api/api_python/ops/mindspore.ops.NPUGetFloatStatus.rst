mindspore.ops.NPUGetFloatStatus
================================

.. py:class:: mindspore.ops.NPUGetFloatStatus

    更新标识，通过执行 :class:`mindspore.ops.NPUAllocFloatStatus` 或取最新溢出状态。

    标志是一个Tensor，其shape为 :math:`(8,)` ，数据类型为 `mindspore.dtype.float32` 。如果标志的和等于0，则没有发生溢出。如果标志之和大于0，则发生溢出。此外，使用有严格的顺序要求，即在使用 :class:`NPUGetFloatStatus` 算子之前，需要确保 :class:`NPUClearFloatStatus` 和需执行的计算已执行。使用 :class:`mindspore.ops.Depend` 确保执行顺序。

    输入：
        - **x** (Tensor) - :class:`NPUAllocFloatStatus` 的输出Tensor。数据类型必须为float16或float32。 :math:`(N,*)` ，其中 :math:`*` 表示任意附加维度，其rank应小于8。

    输出：
        Tensor，shape与 `x` 相同。Tensor中的所有元素都将为零。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
