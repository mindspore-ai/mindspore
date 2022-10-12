mindspore.ops.sequence_mask
============================

.. py:function:: mindspore.ops.sequence_mask(lengths, maxlen=None)

    返回一个表示每个单元的前N个位置的掩码Tensor，内部元素数据类型为bool。

    如果 `lengths` 的shape为 :math:`(d_1, d_2, ..., d_n)` ，则生成的Tensor掩码拥有数据类型，其shape为 :math:`(d_1, d_2, ..., d_n, maxlen)` ，且mask :math:`[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])` 。

    参数：
        - **lengths** (Tensor) - 用来计算掩码的Tensor，一般代表长度。此Tensor中的所有值都应小于或等于 `maxlen` 。大于 `maxlen` 的值将被视为 `maxlen` 。其数据类型为int32或int64。
        - **maxlen** (int) - 指定返回Tensor的长度。其值为正数，且与 `lengths` 中的元素数据类型相同。默认为None。

    返回：
        返回一个Tensor，shape为 `lengths.shape + (maxlen,)` 。

    异常：
        - **TypeError** - `lengths` 不是Tensor。
        - **TypeError** - `maxlen` 不是int。
        - **TypeError** - `lengths` 的数据类型既不是int32，也不是int64。