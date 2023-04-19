mindspore.ops.MatrixDeterminant
===============================

.. py:class:: mindspore.ops.MatrixDeterminant

    计算一个或多个方阵的行列式。

    更多参考详见 :func:`mindspore.ops.det`。

    输入：
        - **input** (Tensor) - 输入Tensor， shape 为 :math:`[..., M, M]` 。矩阵必须至少有两个维度，最后两个维度尺寸必须相同。支持的数据类型为float32、float64、complex64或complex128。

    输出：
        Tensor，其shape为 :math:`input.shape[:-2]` ，数据类型与 `input` 相同。
