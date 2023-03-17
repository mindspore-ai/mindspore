mindspore.ops.MatrixTriangularSolve
===================================

.. py:class:: mindspore.ops.MatrixTriangularSolve(lower=True, adjoint=False)

    返回一个新Tensor，其为含上三角矩阵或下三角矩阵的线性方程组的解。

    .. note::
        仅在GPU上支持广播机制。

    参数：
        - **lower** (bool，可选) - 如果为True， `matrix` 的最内矩阵为下三角矩阵。默认值：True。
        - **adjoint** (bool，可选) - 如果为True，使用 `matrix` 的伴随求解。默认值：False。
  
    输入：
        - **matrix** (Tensor) - Tensor，其shape为 :math:`(*, M, M)` ，类型支持float32、float64、complex64和complex128。
        - **rhs** (Tensor) - Tensor，其shape为 :math:`(*, M, N)` ，类型支持float32、float64、complex64和complex128。

    输出：
        Tensor，其shape为 :math:`(*, M, N)` ，数据类型与 `matrix` 相同。

    异常：
        - **TypeError** - 如果 `matrix` 或 `rhs` 不是Tensor。
        - **TypeError** - 如果 `lower` 或 `adjoint` 不是bool型。
        - **ValueError** - 如果在GPU平台上， `matrix` 和 `rhs` 的batch大小不满足广播条件或者在
          其他平台上 `matrix` 和 `rhs` 的batch大小不相等。
        - **ValueError** - 如果 `matrix` 的最内两维不相等。
        - **ValueError** - 如果 `matrix` 和 `rhs` 的倒数第二维不相等。
