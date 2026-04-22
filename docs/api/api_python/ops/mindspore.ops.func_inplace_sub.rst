mindspore.ops.inplace_sub
=========================

.. py:function:: mindspore.ops.inplace_sub(x, v, indices)

    根据索引将 `x` 减 `v` 。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    对于 `indices` 的每个元素下标 :math:`i, ..., j` ：

    .. math::
        x[\text{index}[i, ..., j]] \mathrel{-}= v[i, ..., j]

    参数：
        - **x** (Tensor) - 输入tensor。
        - **v** (Tensor) - 被 `x` 减去的tensor。
        - **indices** (Union[int, tuple]) - 对输入 `x` 沿第0维度的索引。

    返回：
        Tensor
