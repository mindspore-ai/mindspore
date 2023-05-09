mindspore.ops.matrix_solve
==========================

.. py:function:: mindspore.ops.matrix_solve(matrix, rhs, adjoint=False)

    求解线性方程组。

    .. math::
        \begin{aligned}
        &matrix[..., M, M] * x[..., M, K] = rhs[..., M, K]\\
        &adjoint(matrix[..., M, M]) * x[..., M, K] = rhs[..., M, K]
        \end{aligned}

    .. warning::
        - 当平台为GPU时，如果 `matrix` 中的矩阵不可逆，将产生错误或者返回一个未知结果。

    参数：
        - **matrix** (Tensor) - 输入Tensor， shape 为 :math:`(..., M, M)` 。
        - **rhs** (Tensor) - 输入Tensor， shape 为 :math:`(..., M, K)` 。 `rhs` 的 dtype 必须与 `matrix` 的 dtype 相同。
        - **adjoint** (bool) - 表示是否需要在求解前对输入矩阵 `matrix` 做共轭转置，默认值： ``False`` 。

    返回：
        Tensor，与 `rhs` 的 shape 和数据类型相同。

    异常：
        - **TypeError** -  `adjoint` 不为 bool。
        - **TypeError** -  `matrix` 的 dtype 不属于以下类型： mstype.float16、 mstype.float32、 mstype.float64、 mstype.complex64 和 mstype.complex128。
        - **TypeError** -  `rhs` 的 dtype 与 `matrix` 的 dtype 不相同。
        - **ValueError** - `matrix` 的维度小于2。
        - **ValueError** - `rhs` 的维度与 `matrix` 的维度不相等。
        - **ValueError** - `matrix` 最内侧的两个维度不相等。
        - **ValueError** - `rhs` 最内侧的两个维度和 `matrix` 不能匹配。
        - **ValueError** - `matrix` 中的矩阵不可逆。
