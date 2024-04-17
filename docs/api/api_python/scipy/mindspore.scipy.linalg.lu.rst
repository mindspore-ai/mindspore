mindspore.scipy.linalg.lu
=========================

.. py:function:: mindspore.scipy.linalg.lu(a, permute_l=False, overwrite_a=False, check_finite=True)

    计算通用矩阵的LU分解。

    分解为：

    .. math::
        A = P L U

    其中， :math:`P` 是一个置换矩阵， :math:`L` 是对角线元素全为1的下三角矩阵， :math:`U` 是上三角矩阵。

    .. note::
        - Windows平台上还不支持 `LU`。
        - 仅支持float32、float64、int32、int64类型的Tensor类型。
        - 如果Tensor是int32、int64类型，它将被强制转换为：mstype.float64类型。

    参数：
        - **a** (Tensor) - 要分解的 :math:`(M, N)` 方阵。
          如果输入Tensor不是float类型，那么它将被强制转换为：mstype.float32。
        - **permute_l** (bool, 可选) - 执行乘法运算 :math:`P L`（默认：不进行置换）。
          默认值：``False``。
        - **overwrite_a** (bool, 可选) - 是否覆盖 :math:`a` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **check_finite** (bool, 可选) - 是否检查输入矩阵是否只包含有限数。
          禁用可能会带来性能增益，但如果输入确实包含INF或NaN，则可能会导致问题（崩溃、程序不终止）。
          默认值：``True``。

    返回：
        **如果 permute_l == False**

        - **p** (Tensor) - :math:`(M, M)` 置换矩阵。
        - **l** (Tensor) - :math:`(M, K)` 对角线元素全为1的下三角矩阵或梯形矩阵。 :math:`K = min(M, N)`。
        - **u** (Tensor) - :math:`(K, N)` 上三角矩阵或梯形矩阵。

        **如果 permute_l == True**

        - **pl** (Tensor) - :math:`(M, K)` 置换L矩阵。 :math:`K = min(M,N)`。
        - **u** (Tensor) - :math:`(K, N)` 上三角矩阵或梯形矩阵。
