mindspore.ops.SpaceToBatchND
============================

.. py:class:: mindspore.ops.SpaceToBatchND(block_shape, paddings)

    将空间维度划分为对应大小的块，并在批次维度重排张量。

    此操作将输入的空间维度(Space) [1, ..., M] 划分为形状为 `block_shape` 的块网格，
    并将这些块在批次维度 (Batch，默认是第零维) 中交错排列。
    如此，输出在空间维度上的截面是输入在对应空间维度上截面的一个网格，
    而输出的批次维度的大小为输入的批次维度的大小乘以空间维度分解成块网格的数量。
    在划分成块之前，输入的空间维度会根据 `paddings` 填充零。

    参数：
        - **block_shape** (Union[list(int), tuple(int), int]) - 块形状描述空间维度为分割的个数，取值需大于或者等于1。如果 `block_shape` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_shape` 为整数，那么所有空间维度分割的个数均为 `block_shape` 。在Ascend后端 `M` 必须为2。
        - **paddings** (Union[tuple, list]) - 空间维度的填充大小。包含M个List，每一个List包含2个整形值，且各值须大于或者等于0。 `paddings[i]` 为对空间维度 `i` 的填充，对应输入Tensor的维度 `i+offset` ， `offset` 为空间维度在输入Tensor维度中的偏移量，其中 `offset=N-M` ， `N` 是输入维度数。
          对空间维度i， `input_shape[i+offset]+paddings[i][0]+paddings[i][1]` 必须能被 `block_shape[i]` 整除。

    输入：
        - **input_x** (Tensor) - SpaceToBatchND 的输入，Ascend平台必须为四维。

    输出：
        Tensor，经过划分排列之后的结果。假设输入的shape为 :math:`(n, c_1, ... c_k, w_1, ..., w_M)` ，且算子的属性为
        :math:`block\_shape` 和 :math:`paddings`，
        那么输出的形状为 :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`，
        其中

        .. math::
            \begin{array}{ll} \\
                n' = n*(block\_shape[0]*...*block\_shape[M-1]) \\
                w'_i = (w_i+paddings[i-1][0]+paddings[i-1][1])//block\_shape[i-1]
            \end{array}

    异常：
        - **TypeError** - 如果 `block_shape` 不是 list，tuple 或者 int。
        - **TypeError** - 如果 `paddings` 不是 list 或者 tuple。
        - **ValueError** - 如果当 `block_shape` 为 list 或 tuple， `block_shape` 不是一维。
        - **ValueError** - 如果 Ascend 平台上 `block_shape` 长度不是2。
        - **ValueError** - 如果 `paddings` 的形状不是 (M, 2), 其中 M 为 `block_shape` 的长度。
        - **ValueError** - 如果 `block_shape` 的元素不是大于或者等于一的整数。
        - **ValueError** - 如果 `paddings` 的元素不是非负的整数。
