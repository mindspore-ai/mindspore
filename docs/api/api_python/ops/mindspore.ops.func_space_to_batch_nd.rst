mindspore.ops.space_to_batch_nd
================================

.. py:function:: mindspore.ops.space_to_batch_nd(input_x, block_size, paddings)

    将空间维度划分为对应大小的块，然后在批次维度重排张量。

    此函数将输入的空间维度 [1, ..., M] 划分为形状为 `block_size` 的块网格，并将这些块在批次维度上（默认是第0维）中交错排列。
    输出的张量在空间维度上的截面是输入在对应空间维度上截面的一个网格，而输出的批次维度的大小为空间维度分解成块网格的数量乘以输入的批次维度的大小。
    在划分成块之前，输入的空间维度会根据 `paddings` 填充零。
    如此，假设输入的形状为 :math:`(n, c_1, ... c_k, w_1, ..., w_M)`，则输出的形状为 :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)` ，
    其中

    .. math::
        \begin{array}{ll} \\
            n' = n*(block\_shape[0] * ... * block\_shape[M]) \\
            w'_i = (w_i + paddings[i][0] + paddings[i][1])//block\_shape[i]
        \end{array}

    参数：
        - **input_x** (Tensor) - 输入张量，Ascend平台必须为四维。
        - **block_size** (Union[list(int), tuple(int), int]) - 块形状描述空间维度为分割的个数。如果 `block_size` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_size` 为整数，那么所有空间维度分割的个数均为 `block_size` 。在Ascend后端 `M` 必须为2。
        - **paddings** (Union[tuple, list]) - 空间维度的填充大小。

    返回：
        Tensor，经过划分排列之后的结果。

    异常：
        - **TypeError** - 如果 `block_size` 不是 list, tuple 或者 int。
        - **TypeError** - 如果 `paddings` 不是 list 或者 tuple。
        - **ValueError** - 如果当 `block_size` 为 list 或 tuple， `block_size` 不是一维。
        - **ValueError** - 如果 Ascend 平台上 `block_size` 长度不是2。
        - **ValueError** - 如果 `paddings` 的形状不是 (M, 2), 其中 M 为 `block_size` 的长度。
        - **ValueError** - 如果 `block_size` 的元素不是大于一的整数。
        - **ValueError** - 如果 `paddings` 的元素不是非负的整数。
