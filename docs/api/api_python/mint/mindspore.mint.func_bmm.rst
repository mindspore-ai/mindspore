mindspore.mint.bmm
===========================

.. py:function:: mindspore.mint.bmm(input, mat2)

    对输入的两个tensor执行批量矩阵乘积。

    .. math::
        \text{output} = \text{input} @ \text{mat2}

    `input` 与 `mat2` 仅支持三维。

    参数：
        - **input** (Tensor) - 输入的第一个tensor。
        - **mat2** (Tensor) - 输入的第二个tensor。

    返回：
        Tensor

