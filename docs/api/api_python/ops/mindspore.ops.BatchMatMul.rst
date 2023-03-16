mindspore.ops.BatchMatMul
=========================

.. py:class:: mindspore.ops.BatchMatMul(transpose_a=False, transpose_b=False)

    基于batch维度的两个Tensor的矩阵乘法。

    .. math::
        \text{output}[..., :, :] = \text{matrix}(x[..., :, :]) * \text{matrix}(y[..., :, :])

    两个输入Tensor必须具有相同的秩，并且秩必须不小于 `3`。

    参数：
        - **transpose_a** (bool) - 如果为True，则在乘法之前转置 `x` 的最后两个维度。默认值：False。
        - **transpose_b** (bool) - 如果为True，则在乘法之前转置 `y` 的最后两个维度。默认值：False。

    输入：
        - **x** (Tensor) - 输入相乘的第一个Tensor。其shape为 :math:`(*B, N, C)` ，其中 :math:`*B` 表示批处理大小，可以是多维度， :math:`N` 和 :math:`C` 是最后两个维度的大小。如果 `transpose_a` 为True，则其shape必须为 :math:`(*B, C, N)` 。
        - **y** (Tensor) - 输入相乘的第二个Tensor。Tensor的shape为 :math:`(*B, C, M)` 。如果 `transpose_b` 为True，则其shape必须为 :math:`(*B, M, C)` 。

    输出：
        Tensor，输出Tensor的shape为 :math:`(*B, N, M)` 。

    异常：
        - **TypeError** - `transpose_a` 或 `transpose_b` 不是bool。
        - **ValueError** - `x` 的shape长度不等于 `y` 的shape长度或 `x` 的shape长度小于3。
