mindspore.ops.lu_solve
======================

.. py:function:: mindspore.ops.lu_solve(b, LU_data, LU_pivots)

    给定LU分解结果 :math:`A` 和列向量 :math:`b`，求解线性方程组的解y :math:`Ay = b`。

    一个矩阵的LU分解可以由 :func:`mindspore.scipy.linalg.lu_factor` 得到。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **b** (Tensor) - 上面公式的列向量 `b` 。 `b` 的shape为 :math:`(*, m, k)`，其中 :math:`*` 是batch的维度， 数据类型是float32、float16。
        - **LU_data** (Tensor) - LU分解的结果。shape为 :math:`(*, m, m)` ，其中 :math:`*` 是batch的维度。
          LU分解的结果可以被分成上三角矩阵 U 和下三角矩阵 L , 数据类型为 float32、float16。
        - **LU_pivots** (Tensor) - LU分解的主元。shape为 :math:`(*, m)` ，其中 :math:`*` 是batch的维度。主元可以被转为变换矩阵P， 数据类型为int32。

    返回：
        Tensor，与 `b` 和 `LU_data` 的数据类型相同。

    异常：
        - **TypeError** -  `b` 或 `LU_data` 的 dtype 不属于以下类型： mstype.float16、mstype.float32。
        - **TypeError** -  `LU_pivots` 的 dtype 不属于以下类型： mstype.int32。
        - **TypeError** -  `b` ， `LU_data` 或 `LU_pivots` 不为Tensor。
        - **TypeError** -  `b` 的 dtype 与 `LU_data` 的 dtype 不相同。
        - **ValueError** - `LU_pivots` 的 batch 维度与 `LU_data` 的 batch 维度不相等。
        - **ValueError** - `b` 的维度小于2，`LU_data` 的维度小于2，或 `LU_pivots` 的维度小于1.