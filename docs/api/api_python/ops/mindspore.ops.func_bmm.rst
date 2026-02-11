mindspore.ops.bmm
=================

.. py:function:: mindspore.ops.bmm(input_x, mat2)

    对输入的两个tensor执行批量矩阵乘积。

    .. math::
        \text{output}[..., :, :] = \text{matrix}(input_x[..., :, :]) * \text{matrix}(mat2[..., :, :])

    `input_x` 的维度不能小于 `3` ， `mat2` 的维度不能小于 `2` 。

    参数：
        - **input_x** (Tensor) - 第一个输入tensor。
        - **mat2** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
