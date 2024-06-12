mindspore.mint.bmm
===========================

.. py:function:: mindspore.mint.bmm(input, mat2)

    基于batch维度的两个Tensor的矩阵乘法，仅支持三维。

    .. math::
        \text{output}[b, i, j] = \text{input}[b, i, k] @ \text{mat2}[b, k, j]

    参数：
        - **input** (Tensor) - 输入相乘的第一个Tensor。必须是三维Tensor。
        - **mat2** (Tensor) - 输入相乘的第二个Tensor。必须是三维Tensor。

    返回：
        Tensor，输出Tensor的shape为 :math:`(B, N, M)` 。其中每个矩阵是输入批次中相应矩阵的乘积。

    异常：
        - **ValueError** - `input` 或 `mat2` 的维度不为3。
        - **ValueError** - `input` 第三维的长度不等于 `mat2` 第二维的长度。
        - **ValueError** - `input` 的 batch 维长度不等于 `mat2` 的 batch 维长度。

