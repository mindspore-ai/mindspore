mindspore.ops.extend.bmm
===========================

.. py:function:: mindspore.ops.extend.bmm(input, mat2)

    基于batch维度的两个Tensor的矩阵乘法，仅支持三维。

    .. math::
        \text{output}[:, :, :] = \text{matrix}(input[:, :, :]) * \text{matrix}(mat2[:, :, :])

    `input` 和 `mat2` 的维度只能为 `3`。

    参数：
        - **input** (Tensor) - 输入相乘的第一个Tensor。其shape为 :math:`(B, N, C)` ，其中 :math:`B` 表示批处理大小， :math:`N` 和 :math:`C` 是最后两个维度的大小。
        - **mat2** (Tensor) - 输入相乘的第二个Tensor。Tensor的shape为 :math:`(B, C, M)` 。

    返回：
        Tensor，输出Tensor的shape为 :math:`(B, N, M)` 。

    异常：
        - **ValueError** - `input` 的维度不为 `3`。
        - **ValueError** - `input` 第三维的长度不等于 `mat2` 第二维的长度。

