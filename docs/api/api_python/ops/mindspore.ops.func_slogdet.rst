mindspore.ops.slogdet
=====================

.. py:function:: mindspore.ops.slogdet(input)

    对一个或多个方阵行列式的绝对值取对数，返回其符号和值。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`[..., M, M]` 。矩阵必须至少有两个维度，最后两个维度尺寸必须相同。支持的数据类型为float32、float64、complex64或complex128。

    返回：
        - Tensor，行列式的绝对值的对数的符号，shape为 `input.shape[:-2]` ，数据类型与 `input` 相同。
        - Tensor，行列式的绝对值的对数，shape为 `input.shape[:-2]` ，数据类型与 `input` 相同。

    异常：
        - **TypeError** -  `input` 不为 Tensor。
        - **TypeError** -  `input` 的数据类型不为以下类型：float32、 float64、 complex64 和 complex128。
        - **ValueError** - `input` 的最后两个维度大小不同。
        - **ValueError** - `input` 的维数小于2。
