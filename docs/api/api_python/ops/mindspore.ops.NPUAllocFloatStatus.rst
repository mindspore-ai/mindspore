mindspore.ops.NPUAllocFloatStatus
==================================

.. py:class:: mindspore.ops.NPUAllocFloatStatus

    分配一个标志来存储溢出状态。

    标志是一个Tensor，其shape为 :math:`(8,)` ，数据类型为 `mindspore.dtype.float32` 。

    .. note::
        请参考 :class:`mindspore.ops.NPUGetFloatStatus` 的样例。

    输出：
        Tensor，shape为 :math:`(8,)` 。
