mindspore.ops.Svd
=================

.. py:class:: mindspore.ops.Svd(full_matrices=False, compute_uv=True)

    计算一个或多个矩阵的奇异值分解。

    详情请查看 :func:`mindspore.ops.svd` 。

    参数：
        - **full_matrices** (bool, 可选) - 如果为 ``True`` ，则计算完整的 :math:`U` 和 :math:`V` 。否则仅计算前P个奇异向量，其中，P为M和N中的较小值，M和N分别是输入矩阵的行和列。默认值： ``False`` 。
        - **compute_uv** (bool, 可选) - 如果这个参数为 ``True`` ，则计算 :math:`U` 和 :math:`V` ，否则只计算 :math:`S` 。默认值： ``True`` 。

    输入：
        - **input** (Tensor) - 待分解的矩阵。shape为 :math:`(*, M, N)` ，支持的数据类型为float32和float64。

    输出：
        - **s** (Tensor) - 奇异值。shape为 :math:`(*, P)` 。
        - **u** (Tensor) - 左奇异向量。如果 `compute_uv` 为 ``False`` ，则值为 ``0`` 。shape为 :math:`(*, M, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, M, M)` 。
        - **v** (Tensor) - 右奇异向量。如果 `compute_uv` 为 ``False`` ，则值为 ``0`` 。shape为 :math:`(*, N, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, N, N)` 。
