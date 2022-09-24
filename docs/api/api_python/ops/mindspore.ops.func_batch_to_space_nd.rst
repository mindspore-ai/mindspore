mindspore.ops.batch_to_space_nd
================================

.. py:function:: mindspore.ops.batch_to_space_nd(input_x, block_shape, crops)

    用块划分批次维度，并将这些块交错回空间维度。

    此函数会将批次维度 `N` 划分为具有 `block_shape` 的块，即输出张量的 `N` 维度是划分后对应的块数。
    输出张量的 `H` 、`W` 维度是原始的 `H` 、`W` 维度和 `block_shape` 的乘积从维度裁剪给定。
    如此，若输入的shape为 :math:`(n, c, h, w)` ，则输出的shape为 :math:`(n', c', h', w')` 。
    其中

    :math:`n' = n//(block\_shape[0]*block\_shape[1])`

    :math:`c' = c`

    :math:`h' = h*block\_shape[0]-crops[0][0]-crops[0][1]`

    :math:`w' = w*block\_shape[1]-crops[1][0]-crops[1][1]`

    参数：
        - **input_x** (Tensor) - 输入张量，必须大于或者等于四维（Ascend平台必须为4维）。批次维度需能被 `block_shape` 整除。支持数据类型float16和float32。
        - **block_shape** (Union[list(int), tuple(int), int]) - 分割批次维度的块的数量，取值需大于1。如果 `block_shape` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_shape` 为整数，那么所有空间维度分割的个数均为 `block_shape` 。 `M` 必须为2。
        - **crops** (Union[list(int), tuple(int)]) - 空间维度的裁剪大小，包含两个长度为2的list，分别对应空间维度H和W。取值需大于或等于0，同时要求 `input_shape[i+2] * block_shape[i] > crops[i][0] + crops[i][1]` 。

    返回：
        Tensor，经过划分排列之后的结果。

    异常：
        - **TypeError** - 如果 `block_shape` 不是 list、tuple 或者 int。
        - **TypeError** - 如果 `crops` 不是 list 或者 tuple。
        - **ValueError** - 如果当 `block_shape` 为 list 或 tuple， `block_shape` 不是一维。
        - **ValueError** - 如果 `block_shape` 或 `crops` 长度不是2。
        - **ValueError** - 如果 `block_shape` 的元素不是大于一的整数。
        - **ValueError** - 如果 `crops` 的形状不是 (M, 2)，其中 M 为 `block_shape` 的长度。
        - **ValueError** - 如果 `crops` 的元素不是非负的整数。
