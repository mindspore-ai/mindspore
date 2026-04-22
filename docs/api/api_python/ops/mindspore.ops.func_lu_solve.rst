mindspore.ops.lu_solve
======================

.. py:function:: mindspore.ops.lu_solve(b, LU_data, LU_pivots)

    计算线性方程组 :math:`Ay = b` 的LU解。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `b` 将重命名为 `input`，并新增 `left` 和 `adjoint` 可选参数。

    .. note::
        - `b` 的shape为 :math:`(*, m, k)` ， `LU_data` 的shape为 :math:`(*, m, m)` ，
          `LU_pivots` 的shape为 :math:`(*, m)` ，:math:`*` 表示batch数量。

    参数：
        - **b** (Tensor) - 上面公式的列向量 `b` 。
        - **LU_data** (Tensor) - LU分解的结果，上面公式中的 `A` 。
        - **LU_pivots** (Tensor) - LU分解的主元，主元可以被转为变换矩阵P。

    返回：
        Tensor
