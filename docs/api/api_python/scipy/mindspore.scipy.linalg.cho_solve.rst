mindspore.scipy.linalg.cho_solve
================================

.. py:function:: mindspore.scipy.linalg.cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True)

    给定 :math:`A` 的cholesky分解，求解线性方程组。

    .. math::
        A x = b

    .. note::
        - 仅支持float32、float64、int32、int64类型的Tensor类型。
        - 如果Tensor是int32、int64类型，它将被强制转换为：mstype.float64类型。

    参数：
        - **c_and_lower** ((Tensor, bool)) - :math:`a` 的cholesky分解，由：:func:`mindspore.scipy.linalg.cho_factor` 计算得出。
        - **b** (Tensor) - 方程右侧的值。
        - **overwrite_b** (bool, 可选) - 是否覆盖：:math:`b` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **check_finite** (bool, 可选) - 是否检查输入矩阵是否只包含有限数。
          禁用可能会带来性能增益，但如果输入确实包含INF或NaN，则可能会导致问题（崩溃、程序不终止）。
          默认值：``True``。

    返回：
        Tensor，线性方程 :math:`A x = b` 的解。
