mindspore.ops.stack
====================

.. py:function:: mindspore.ops.stack(tensors, axis=0)

    在指定轴上对输入的tensors进行堆叠。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`axis` 将改为 `dim`。

    参数：
        - **tensors** (Union[tuple, list]) - 输入tensors。
        - **axis** (int) - 指定轴，默认 ``0`` 。

    返回：
        Tensor
