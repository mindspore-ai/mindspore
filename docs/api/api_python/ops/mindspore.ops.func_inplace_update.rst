mindspore.ops.inplace_update
============================

.. py:function:: mindspore.ops.inplace_update(x, v, indices)

    根据索引将 `x` 更新为 `v` 。

    .. warning::
        - 该接口已废弃，2.9.0（不含）之后版本将删除。
        - 这是一个实验性API，后续可能修改或删除。

    对于 `indices` 的每个元素下标 :math:`i, ..., j` ：

    .. math::
        x[\text{index}[i, ..., j]] = v[i, ..., j]

    参数：
        - **x** (Tensor) - 输入tensor。
        - **v** (Tensor) - 更新的tensor。
        - **indices** (Union[int, tuple[int], Tensor]) - 对输入 `x` 沿第0维度的索引。

    返回：
        Tensor
