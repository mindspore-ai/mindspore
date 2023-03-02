mindspore.ops.NPUClearFloatStatus
=================================

.. py:class:: mindspore.ops.NPUClearFloatStatus

    清除存储溢出状态的标识。

    .. note::
        该标志位于 `Acend` 设备上的寄存器中。在调用 `NPUClearFloatStatus` 后，它将被重置，不能再次重用。此外，使用有严格的使用顺序要求，即在使用 :class:`mindspore.ops.NPUGetFloatStatus` 算子之前，需要确保 `NPUClearFloatStatus` 和需执行的计算已执行。我们使用 :class:`mindspore.ops.Depend` 确保执行顺序。

        请参考 :class:`mindspore.ops.NPUGetFloatStatus` 的样例。

    输入：
        - **x** (Tensor) - :class:`mindspore.ops.NPUAllocFloatStatus` 的输出Tensor。数据类型必须为float16或float32。
        
    输出：
        Tensor，shape与 `x` 相同。Tensor中的所有元素都将为零。
