mindspore.ops.cat
==================

.. py:function:: mindspore.ops.cat(tensors, axis=0)

    在指定轴上拼接输入tensor。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `axis` 将重命名为 `dim` 。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 输入tensors。除了指定的拼接轴 `axis` 之外，其他轴的shape都应相等。
        - **axis** (int) - 指定轴。默认 ``0`` 。

    返回：
        Tensor
