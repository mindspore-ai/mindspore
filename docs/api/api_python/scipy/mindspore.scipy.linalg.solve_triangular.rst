mindspore.scipy.linalg.solve_triangular
=======================================

.. py:function:: mindspore.scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=None, check_finite=True)

    假设 `a` 是一个三角矩阵，求等式系统 :math:`a x = b` 的解 `x` 。

    .. note::
        - `solve_triangular` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `solve_triangular` 尚不支持Windows平台。

    参数：
        - **a** (Tensor) - shape为 :math:`(*, M, M)` 的三角方阵，其中 :math:`*` 表示零或者更多的Batch维度。
        - **b** (Tensor) - shape为 :math:`(*, M, N)` 或者 :math:`(*, M)` 的矩阵或向量。
        - **trans** (Union[int, str], 可选) - 求解系统的类型，默认值： ``0`` 。

          ========  =========
          trans值     求解系统
          ========  =========
          0 或 'N'  a x  = b
          1 或 'T'  a^T x = b
          2 或 'C'  a^H x = b
          ========  =========

        - **lower** (bool, 可选) - `a` 是否为下三角矩阵，默认值： ``False`` 。
        - **unit_diagonal** (bool, 可选) - 如果为 ``True``，那么 `a` 的对角线会被全部置为1，默认值： ``False`` 。
        - **overwrite_b** (bool, 可选) - 在MindSpore中，当前这个参数不起作用。默认值： ``False`` 。
        - **debug** (Any, 可选) - 在MindSpore中，当前这个参数不起作用。默认值： ``None`` 。
        - **check_finite** (bool, 可选) - 在MindSpore中，当前这个参数不起作用。默认值： ``True`` 。

    返回：
        Tensor，shape为 :math:`(*, M, N)` 或者 :math:`(*, M)` 的矩阵。:math:`a x = b` 等式的解，其中 `x` 的shape与 `b` 的shape相同。

    异常：
        - **ValueError** - 如果 `a` 的维度小于2。
        - **ValueError** - 如果 `a` 不是方阵。
        - **TypeError** - 如果 `a` 和 `b` 的数据类型不同。
        - **ValueError** - 如果 `a` 和 `b` 的shape不匹配。
        - **ValueError** - 如果 `trans` 不是 ``0`` 、 ``1``、 ``2``、 ``'N'``、 ``'T'`` 或 ``'C'`` 。