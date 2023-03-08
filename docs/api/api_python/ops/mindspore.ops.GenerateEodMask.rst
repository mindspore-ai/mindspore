mindspore.ops.GenerateEodMask
=============================

.. py:class:: mindspore.ops.GenerateEodMask(eod_token_id)

    根据输入的 `inputs_ids`， 遇到 `eod_token_id` 时，会将输出的位置编码和注意力编码全部重置。
    即`position_id`从0开始重新计数,同时对应的掩码矩阵也会填充为0。

    参数：
        - **eod_token_id** (int) - `eod_token_id` 的数值。在NLP场景中，这个值对应词表中的 `EodOfDocument` 的符号编码。

    输入：
        - **inputs_ids** (Tensor) - 词序列。是一个二维Tensor，其shape为 :math:`(batch\_size, seq\_length)` 。

    输出：
        - **position_id** (Tensor) - 位置编码矩阵。数据类型和shape与输入 `inputs_ids` 相同。
        - **attention_mask** (Tensor) - 注意力掩码矩阵。类型为float16，其shape为： :math:`(batch\_size, seq\_length)` 。

    异常：
        - **TypeError** - 如果 `eod_token_id` 的数据类型不是int。
        - **TypeError** - 如果 `inputs_ids` 的数据类型不是整数类型。
        - **ValueError** - 如果 `inputs_ids` 不是二维的Tensor。
