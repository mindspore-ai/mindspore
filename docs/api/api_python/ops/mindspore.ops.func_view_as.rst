mindspore.ops.view_as
======================

.. py:function:: mindspore.ops.view_as(input, other)

    根据 `other` 的shape改变输入Tensor的shape。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **other** (Tensor) - 返回Tensor的shape与 `other` 的shape一致。

    返回：
        Tensor，和 `other` 具有相同的shape。

    异常：
        - **TypeError** - 输入非Tensor。
