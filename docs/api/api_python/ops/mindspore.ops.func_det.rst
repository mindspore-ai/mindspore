mindspore.ops.det
=================

.. py:function:: mindspore.ops.det(x)

    计算一个或多个方阵的行列式。

    参数：
        - **x** (Tensor) - 输入Tensor， shape 为 :math:`[..., M, M]` 。矩阵必须至少有两个维度，最后两个维度尺寸必须相同。支持的数据类型为float32、float64、complex64或complex128。

    返回：
        Tensor，形状为 `x.shape[:-2]` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** -  `x` 不为 Tensor。
        - **TypeError** -  `x` 的数据类型不为以下类型：float32、float64、complex64 和 complex128。
        - **ValueError** - `x` 的最后两个维度大小不同。
        - **ValueError** - `x` 的维数小于2。
