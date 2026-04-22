mindspore.ops.batch_to_space_nd
================================

.. py:function:: mindspore.ops.batch_to_space_nd(input_x, block_shape, crops)

    用块划分批次维度，并将这些块交错回空间维度。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    此函数会将批次维度 `N` 划分为具有 `block_shape` 的块，即输出张量的 `N` 维度是划分后对应的块数。
    输出张量的 :math:`w_1, ..., w_M` 维度是原始的 :math:`w_1, ..., w_M` 维度和 `block_shape` 的乘积从维度裁剪给定。
    若输入的shape为 :math:`(n, c_1, ... c_k, w_1, ..., w_M)`，则输出的shape为 :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)` 。
    其中，

    .. math::
            \begin{array}{ll} \\
                n' = n//(block\_shape[0]*...*block\_shape[M-1]) \\
                w'_i = w_i*block\_shape[i-1]-crops[i-1][0]-crops[i-1][1]
            \end{array}

    参数：
        - **input_x** (Tensor) - 输入tensor，必须大于或者等于二维（Ascend平台必须为四维）。批次维度需能被 `block_shape` 整除。
        - **block_shape** (Union[list(int), tuple(int), int]) - 分割批次维度的块的数量，取值需大于或者等于1。如果 `block_shape` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_shape` 为整数，那么所有空间维度分割的个数均为 `block_shape` 。在Ascend后端 `M` 必须为2。
        - **crops** (Union[list(int), tuple(int)]) - 空间维度的裁剪大小，包含 `M` 个长度为2的list，取值需大于或等于0。`crops[i]` 为对空间维度 `i` 的填充，对应输入tensor的维度 `i+offset` ， `offset` 为空间维度在输入tensor维度中的偏移量，其中 `offset=N-M` ， `N` 是输入维度数。同时要求 :math:`input\_shape[i+offset]*block\_shape[i] > crops[i][0]+crops[i][1]` 。

    返回：
        Tensor
