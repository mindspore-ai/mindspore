mindspore.ops.log_matrix_determinant
====================================

.. py:function:: mindspore.ops.log_matrix_determinant(x)

    对一个或多个方阵行列式的绝对值取对数，返回其符号和值。

    参数：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`[..., M, M]` 。矩阵必须至少有两个维度，最后两个维度尺寸必须相同。支持的数据类型为float32、float64、complex64或complex128。

    返回：
        Tensor，行列式的绝对值的对数的符号，shape为 `x.shape[:-2]` ，数据类型与 `x` 相同。

        Tensor，行列式的绝对值的对数，shape为 `x.shape[:-2]` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** -  `x` 不为 Tensor。
        - **TypeError** -  `x` 的数据类型不为以下类型：float32、 float64、 complex64 和 complex128。
        - **ValueError** - `x` 的最后两个维度大小不同。
        - **ValueError** - `x` 的维数小于2。
