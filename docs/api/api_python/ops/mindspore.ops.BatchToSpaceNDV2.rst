mindspore.ops.BatchToSpaceNDV2
==============================

.. py:class:: mindspore.ops.BatchToSpaceNDV2

    用块划分批次维度，并将这些块交错回空间维度。

    更多参考详见 :func:`mindspore.ops.batch_to_space_nd`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **input_x** (Tensor) - 输入Tensor，必须大于或者等于二维（Ascend平台必须为四维）。批次维度需能被 `block_shape` 整除。
        - **block_shape** (Tensor) - 分割批次维度的块的数量，取值需大于或者等于1。如果 `block_shape` 为list或者tuple，其长度 `M` 为空间维度的长度。如果 `block_shape` 为整数，那么所有空间维度分割的个数均为 `block_shape` 。在Ascend后端 `M` 必须为2。
        - **crops** (Tensor) - 空间维度的裁剪大小，包含 `M` 个长度为2的list，取值需大于或等于0。`crops[i]` 为对空间维度 `i` 的填充，对应输入Tensor的维度 `i+offset` ， `offset` 为空间维度在输入Tensor维度中的偏移量，其中 `offset=N-M` ， `N` 是输入维度数。同时要求 `input_shape[i+offset] * block_shape[i] > crops[i][0] + crops[i][1]` 。

    输出：
        Tensor，包含Tensor经过划分batch维并重新排列之后的结果。
