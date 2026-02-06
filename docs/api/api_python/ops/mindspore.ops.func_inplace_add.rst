mindspore.ops.inplace_add
=========================

.. py:function:: mindspore.ops.inplace_add(x, v, indices)

    根据索引将 `x` 加到 `v` 中。

    对于 `indices` 的每个元素下标 :math:`i, ..., j` ：

    .. math::
        x[\text{index}[i, ..., j]] \mathrel{+}= v[i, ..., j]

    参数：
        - **x** (Tensor) - 输入tensor。
        - **v** (Tensor) - 与 `x` 相加的tensor。
        - **indices** (Union[int, tuple]) - 对输入 `x` 沿第零维度的索引。

    返回：
        Tensor
