mindspore.ops.svd
==================

.. py:function:: mindspore.ops.svd(input, full_matrices=False, compute_uv=True)

    计算单个或多个矩阵的奇异值分解。

    设矩阵 :math:`A` ，svd返回奇异值 :math:`S` 、左奇异向量 :math:`U` 和右奇异向量 :math:`V` 。满足以下公式:

    .. math::
        A=U*diag(S)*V^{T}

    参数：
        - **input** (Tensor) - 待分解的矩阵。shape为 :math:`(*, M, N)` ，支持的数据类型为float32和float64。
        - **full_matrices** (bool, 可选) - 如果为True，则计算完整的 :math:`U` 和 :math:`V` 。否则仅计算前P个奇异向量，P为M和N中的较小值，M和N分别是输入矩阵的行和列。默认值：False。
        - **compute_uv** (bool, 可选) - 如果这个参数为True，则计算 :math:`U` 和 :math:`V` ，否则只计算 :math:`S` 。默认值：True。
    返回：
        - **s** (Tensor) - 奇异值。shape为 :math:`(*, P)` 。
        - **u** (Tensor) - 左奇异向量。如果 `compute_uv` 为False，该值不会返回。shape为 :math:`(*, M, P)` 。如果 `full_matrices` 为True，则shape为 :math:`(*, M, M)` 。
        - **v** (Tensor) - 右奇异向量。如果 `compute_uv` 为False，该值不会返回。shape为 :math:`(*, N, P)` 。如果 `full_matrices` 为True，则shape为 :math:`(*, N, N)` 。

    异常：
        - **TypeError** - `full_matrices` 或 `compute_uv` 不是bool类型。
        - **TypeError** - 输入的rank小于2。
        - **TypeError** - 输入的数据类型不为float32或float64。
