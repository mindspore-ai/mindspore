mindspore.ops.matrix_solve
==========================

.. py:function:: mindspore.ops.matrix_solve(matrix, rhs, adjoint=False)

    求解线性方程组。

    .. warning::
        - 该接口已废弃，2.9.0（不含）之后版本将删除。
        - 当平台为GPU时，如果 `matrix` 中的矩阵不可逆，将产生错误或者返回一个未知结果。

    .. math::
        \begin{aligned}
        &matrix[..., M, M] * x[..., M, K] = rhs[..., M, K]\\
        &adjoint(matrix[..., M, M]) * x[..., M, K] = rhs[..., M, K]
        \end{aligned}

    参数：
        - **matrix** (Tensor) - 第一个输入tensor。
        - **rhs** (Tensor) - 第二个输入tensor。
        - **adjoint** (bool，可选) - 表示是否需要在求解前对输入矩阵 `matrix` 做共轭转置，默认 ``False`` 。

    返回：
        Tensor
