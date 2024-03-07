mindspore.scipy.linalg.lu_factor
================================

.. py:function:: mindspore.scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)

    计算方阵的LU分解，其输出可以直接作为 `lu_solve` 的输入。

    分解为：

    .. math::
        a = P L U

    其中， :math:`P` 是一个置换矩阵， :math:`L` 是对角线元素全为1的下三角矩阵， :math:`U` 是上三角矩阵。

    .. note::
        - Windows平台上还不支持 `lu_factor`。
        - 仅支持float32、float64、int32、int64的Tensor类型。
        - 如果Tensor是int32、int64类型，它将被强制转换为：mstype.float64类型。

    参数：
        - **a** (Tensor) - 要分解的 :math:`(M, M)` 方阵。
          如果输入Tensor不是float类型，那么它将被强制转换为：mstype.float32。
        - **overwrite_a** (bool, 可选) - 是否覆盖 :math:`a` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **check_finite** (bool, 可选) - 是否检查输入矩阵是否只包含有限数。
          禁用可能会带来性能增益，但如果输入确实包含 `INF` 或 `NaN`，则可能会导致问题（崩溃、程序不终止）。
          默认值：``True``。

    返回：
        - **lu** (Tensor) - 一个 :math:`(M, M)` 的方阵，在它的上三角中包含 `u`，它的下三角形中包含 `l`。
          不含 `l` 中对角线全为1的元素。
        - **piv** (Tensor) - shape为 :math:`(M,)` 的Tensor，表示置换矩阵 `p` 的索引：索引中的第 `i` 个元素值 `j` 表示矩阵的第 `i` 行与第 `j` 行互换。

    异常：
        - **ValueError** - 如果 :math:`a` 不是2D方阵。
