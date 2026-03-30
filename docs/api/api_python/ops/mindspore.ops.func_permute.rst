mindspore.ops.permute
=====================

.. py:function:: mindspore.ops.permute(input, axis)

    按照指定轴重排输入tensor。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`axis` 将改为 `dims`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (tuple(int)) - 指定顺序的轴。

    返回：
        Tensor
