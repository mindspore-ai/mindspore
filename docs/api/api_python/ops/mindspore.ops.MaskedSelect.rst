mindspore.ops.MaskedSelect
===========================

.. py:class:: mindspore.ops.MaskedSelect

    使用布尔掩码对输入进行选择得到一个新的一维Tensor。掩码Tensor和输入Tensor的shape不需要匹配，但需支持广播。

    **输入：**

    - **x** (Tensor) - 需要进行索引操作的输入Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。
    - **mask** (Tensor[bool]) - 要进行索引的布尔掩码，其shape为 :math:`(x_1, x_2, ..., x_R)` 。

    **输出：**

    一维Tensor，数据类型与x相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。