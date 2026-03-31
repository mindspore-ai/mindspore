mindspore.ops.gather_nd
=======================

.. py:function:: mindspore.ops.gather_nd(input_x, indices)

    根据指定索引获取输入tensor的切片。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    假设 `indices` 是一个K维的整型张量，遵循公式如下：

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    需满足 :math:`indices.shape[-1] <= len(input\_x.shape)` 。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **indices** (Tensor) - 指定索引。

    返回：
        Tensor
