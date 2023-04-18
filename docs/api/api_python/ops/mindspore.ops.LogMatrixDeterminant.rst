mindspore.ops.LogMatrixDeterminant
==================================

.. py:class:: mindspore.ops.LogMatrixDeterminant

    计算一个或多个方块矩阵行列式绝对值的符号和对数。

    更多参考详见 :func:`mindspore.ops.slogdet`。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(..., M, M)` 。矩阵必须至少有两个维度，最后两个维度尺寸必须相同。支持的数据类型为float32、float64、complex64或complex128。

    输出：
        - **sign** (Tensor) - 行列式的绝对值的对数的符号，shape为 :math:`input.shape[:-2]` ，数据类型与 `input` 相同。
        - **y** (Tensor) - 行列式的绝对值的对数，shape为 :math:`input.shape[:-2]` ，数据类型与 `input` 相同。
