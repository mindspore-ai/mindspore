mindspore.ops.MatrixSolve
=========================

.. py:class:: mindspore.ops.MatrixSolve(adjoint=False)

    求解线性方程组。

    参数：
        - **adjoint** (bool，可选) - 指明是用矩阵求解还是用其（逐块）伴随求解。默认值：False。
  
    输入：
        - **matrix** (Tensor) - Tensor，线性方程组系数组成的矩阵，其shape为 :math:`[..., M, M]` 。
        - **rhs** (Tensor) - Tensor，线性方程组结果值组成的矩阵，其shape为 :math:`[..., M, K]` ， `rhs` 与 `matrix` 的类型必须一致。

    输出：
        Tensor，线性方程组解组成的矩阵，与 `rhs` 的shape及类型均相同。

    异常：
        - **TypeError** - 如果 `adjoint` 不是bool型。
        - **TypeError** - 如果 `matrix` 的类型不是以下之一：
          mstype.float16、mstype.float32、mstype.float64、mstype.complex64、mstype.complex128。
        - **TypeError** - 如果 `rhs` 与 `matrix` 的类型不一致。
        - **ValueError** - 如果 `matrix` 的秩小于2。
        - **ValueError** - 如果 `matrix` 和 `rhs` 的维度不同。
        - **ValueError** - 如果 `matrix` 的最内两维不同。
        - **ValueError** - 如果 `rhs` 的最内两维与 `matrix` 不匹配。
