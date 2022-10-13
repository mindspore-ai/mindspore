mindspore.ops.bmm
=================

.. py:function:: mindspore.ops.bmm(input_x, mat2)

    基于batch维度的两个Tensor的矩阵乘法。

    .. math::
        \text{output}[..., :, :] = \text{matrix}(x[..., :, :]) * \text{matrix}(y[..., :, :])

    第一个输入Tensor必不能小于 `3`，第二个输入必不能小于 `2`。

    参数：
        - **input_x** (Tensor) - 输入相乘的第一个Tensor。其shape为 :math:`(*B, N, C)` ，其中 :math:`*B` 表示批处理大小，可以是多维度， :math:`N` 和 :math:`C` 是最后两个维度的大小。
        - **mat2** (Tensor) - 输入相乘的第二个Tensor。Tensor的shape为 :math:`(*B, C, M)` 。

    返回：
        Tensor，输出Tensor的shape为 :math:`(*B, N, M)` 。

    异常：
        - **ValueError** - `input_x` 的shape长度不等于 `mat2` 的shape长度或 `input_x` 的shape长度小于 `3` 。
