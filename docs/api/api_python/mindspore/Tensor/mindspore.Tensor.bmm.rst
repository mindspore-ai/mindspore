mindspore.Tensor.bmm
====================

.. py:method:: mindspore.Tensor.bmm(mat2)

    基于batch维度的两个Tensor的矩阵乘法。

    .. math::
        \text{output}[..., :, :] = \text{matrix}(input_x[..., :, :]) * \text{matrix}(mat2[..., :, :])

    `input_x` 的shape长度必须不小于 `3` ， `mat2` 的shape长度必须小于 `2` 。

    参数：
        - **mat2** (Tensor) - 要相乘的张量，shape大小为 :math:`(*B, C, M)` 。

    返回：
        Tensor，输出Tensor的shape为 :math:`(*B, N, M)` 。

    异常：
        - **ValueError** - `input_x` 的shape长度不等于 `mat2` 的shape长度或 `input_x` 的shape长度小于 `3` 。
