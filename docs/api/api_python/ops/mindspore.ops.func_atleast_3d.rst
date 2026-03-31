mindspore.ops.atleast_3d
=========================

.. py:function:: mindspore.ops.atleast_3d(inputs)

    当输入中的tensor小于三维时，返回它的三维tensor。其他维度的tensor，直接返回原tensor。

    .. warning::
        2.9.0（不含）之后版本中，入参形式将从单个 `inputs` 改为可变参 `*tensors`。

    参数：
        - **inputs** (Union[Tensor, list[Tensor]]) - 输入tensor或tensor列表。

    返回：
        tensor或tensor列表。
