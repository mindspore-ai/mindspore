mindspore.Tensor.svd
====================

.. py:method:: mindspore.Tensor.svd(full_matrices=False, compute_uv=True)

    计算单个或多个矩阵的奇异值分解。

    更多参考详见 :func:`mindspore.ops.svd`。

    参数：
        - **full_matrices** (bool, 可选) - 如果这个参数为True，则计算完整的 :math:`U` 和 :math:`V` 。否则 :math:`U` 和 :math:`V` 的shape和P有关。P是M和N的较小值。M和N是输入矩阵的行和列。默认值：False。
        - **compute_uv** (bool, 可选) - 如果这个参数为True，则计算 :math:`U` 和 :math:`V` 。如果为false，只计算 :math:`S` 。默认值：True。

    返回：
        - **s** (Tensor) - 奇异值。shape为 :math:`(*, P)`。
        - **u** (Tensor) - 左奇异向量。如果compute_uv为False，该值不会返回。shape为 :math:`(*, M, P)` 。如果full_matrices为true，则shape为 :math:`(*, M, M)` 。
        - **v** (Tensor) - 右奇异向量。如果compute_uv为False，该值不会返回。shape为 :math:`(*, P, N)` 。如果full_matrices为true，则shape为 :math:`(*, N, N)` 。

    异常：
        - **TypeError** - `full_matrices` 或 `compute_uv` 不是bool类型。
        - **TypeError** - 输入的rank小于2。
        - **TypeError** - 输入的数据类型不为float32或float64。