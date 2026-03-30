mindspore.ops.space_to_batch_nd
================================

.. py:function:: mindspore.ops.space_to_batch_nd(input_x, block_size, paddings)

    将空间维度分块，并在批次维度重排tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. math::
        \begin{array}{ll} \\
            n' = n*(block\_size[0] * ... * block\_size[M]) \\
            w'_i = (w_i + paddings[i][0] + paddings[i][1])//block\_size[i]
        \end{array}

    .. note::
        - 此函数将输入的空间维度 [1, ..., M] 按 `block_size` 拆分成小块，并重排至批次维度（默认第 0 维）。在分块前，输入的空间维度会根据 `paddings` 填充零。
        - 若输入的形状为 :math:`(n, c_1, ... c_k, w_1, ..., w_M)`，则输出的形状为 :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)` 。
        - 如果 `block_size` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_size` 为整数，那么所有空间维度分割的个数均为 `block_size` 。在Ascend平台 `M` 必须为2。

    参数：
        - **input_x** (Tensor) - 输入张量。
        - **block_size** (Union[list(int), tuple(int), int]) - 空间维度分块大小。
        - **paddings** (Union[tuple, list]) - 空间维度的填充大小。

    返回：
        Tensor
