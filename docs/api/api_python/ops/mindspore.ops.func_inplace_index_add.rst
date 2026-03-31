mindspore.ops.inplace_index_add
===============================

.. py:function:: mindspore.ops.inplace_index_add(var, indices, updates, axis)

    根据指定轴和索引在输入 `var` 中的对应位置加 `updates` 。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    对于 `indices` 的每个元素下标 :math:`i, ..., j` ：

    .. math::
        x[:, \text{indices}[i, ..., j], :] \mathrel{+}= v[:, i, ..., j, :]

    其中 `i` 是 `indices` 中元素的下标， `indices[i]` 所在的轴由输入 `axis` 决定。

    参数：
        - **var** (Union[Parameter, Tensor]) - 输入的parameter或tensor。
        - **indices** (Tensor) - 指定索引，一维tensor。
        - **updates** (Tensor) - 与 `var` 相加的tensor。
        - **axis** (int) - 指定轴。

    返回：
        Tensor
