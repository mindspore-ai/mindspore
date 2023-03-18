mindspore.ops.LuUnpack
======================

.. py:class:: mindspore.ops.LuUnpack(unpack_data=True, unpack_pivots=True)

    将 `LU_data` 和 `LU_pivots` 还原为为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。通常情况下， `LU_data` 和 `LU_pivots` 是矩阵通过LU分解生成的。

    更多参考详见 :func:`mindspore.ops.lu_unpack`。
