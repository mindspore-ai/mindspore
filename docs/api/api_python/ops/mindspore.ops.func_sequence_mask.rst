mindspore.ops.sequence_mask
============================

.. py:function:: mindspore.ops.sequence_mask(lengths, maxlen=None)

    返回一个表示每个单元的前N个位置的掩码tensor，内部元素数据类型为bool。

    如果 `lengths` 的shape为 :math:`(d_1, d_2, ..., d_n)` ，则生成的tensor掩码拥有数据类型，其shape为 :math:`(d_1, d_2, ..., d_n, maxlen)` 。
    且mask :math:`[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])` 。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **lengths** (Tensor) - 用于计算掩码tensor。此tensor中的所有值都应小于或等于 `maxlen` ，大于 `maxlen` 的值将被视为 `maxlen` 。
        - **maxlen** (int) - 返回tensor最后一维的尺寸。必须为正数，且数据类型与 `lengths` 中的元素类型相同。默认 ``None`` 。

    返回：
        Tensor，shape为 `lengths.shape + (maxlen,)` 。
