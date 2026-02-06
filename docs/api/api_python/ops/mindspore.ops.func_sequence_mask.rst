mindspore.ops.sequence_mask
============================

.. py:function:: mindspore.ops.sequence_mask(lengths, maxlen=None)

    返回一个表示每个单元的前N个位置的掩码tensor，内部元素数据类型为bool。

    如果 `lengths` 的shape为 :math:`(d_1, d_2, ..., d_n)` ，则生成的tensor掩码拥有数据类型，其shape为 :math:`(d_1, d_2, ..., d_n, maxlen)` 。
    且mask :math:`[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])` 。

    参数：
        - **lengths** (Tensor) - 输入tensor。此tensor中的所有值都应小于或等于 `maxlen` ，大于 `maxlen` 的值将被视为 `maxlen` 。
        - **maxlen** (int) - 指定返回tensor的长度。默认 ``None`` 。

    返回：
        Tensor，shape为 `lengths.shape + (maxlen,)` 。