mindspore.Tensor.matrix_determinant
===================================

.. py:method:: mindspore.Tensor.matrix_determinant()

    计算一个或多个方阵的行列式。

    `x` 指的当前 Tensor。

    返回：
        Tensor，形状为 `x_shape[:-2]` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** -  `x` 不为 Tensor。
        - **TypeError** -  `x` 的数据类型不为以下类型： mstype.float32、 mstype.float64、 mstype.complex64 和 mstype.complex128。
        - **ValueError** - `x` 的最后两个维度大小不同。
        - **ValueError** - `x` 的维数小于2。