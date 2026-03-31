mindspore.ops.svd
==================

.. py:function:: mindspore.ops.svd(input, full_matrices=False, compute_uv=True)

    计算单个或多个矩阵的奇异值分解。

    .. warning::
        从2.9.0（不含）之后版本开始，签名将调整为 ``svd(input, some=True, compute_uv=True)`` ，当前 `full_matrices` 将替换为 `some`，参数语义不兼容（`full_matrices=False` 对应 `some=True`）。

    设矩阵 :math:`A` ，svd返回奇异值 :math:`S` 、左奇异向量 :math:`U` 和右奇异向量 :math:`V` 。满足以下公式：

    .. math::
        A=U*diag(S)*V^{T}

    参数：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(*, M, N)` 。
        - **full_matrices** (bool, 可选) - 如果为 ``True`` ，则计算完整的 :math:`U` 和 :math:`V` 。否则仅计算前P个奇异向量，P为M和N中的较小值。默认 ``False`` 。
        - **compute_uv** (bool, 可选) - 如果为 ``True`` ，则计算 :math:`U` 和 :math:`V` ，否则只计算 :math:`S` 。默认 ``True`` 。

    返回：
        如果 `compute_uv` 为 ``True`` ，返回三个tensor组成的tuple( `s` , `u` , `v`)。否则，返回单个tensor -> `s` 。

        - `s` 是奇异值tensor。shape为 :math:`(*, P)` 。
        - `u` 是左奇异tensor。如果 `compute_uv` 为 ``False`` ，该值不会返回。shape为 :math:`(*, M, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, M, M)` 。
        - `v` 是右奇异tensor。如果 `compute_uv` 为 ``False`` ，该值不会返回。shape为 :math:`(*, N, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, N, N)` 。
