mindspore.ops.LuUnpack
======================

.. py:class:: mindspore.ops.LuUnpack(unpack_data=True, unpack_pivots=True)

    将 `LU_data` 和 `LU_pivots` 还原为为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。通常情况下， `LU_data` 和 `LU_pivots` 是矩阵通过LU分解生成的。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.lu_unpack`。

    参数：
        - **unpack_data** (bool，可选) - 是否解压缩 `LU_data` 的标志。如果为 ``False`` ，则返回的L和U为 ``None`` 。默认值： ``True`` 。
        - **unpack_pivots** (bool，可选) - 是否将 `LU_pivots` 解压缩为置换矩阵P的标志。如果为 ``False`` ，则返回的P为 ``None`` 。默认值： ``True`` 。

    输入：
        - **LU_data** (Tensor) - Tensor，打包的LU分解数据，shape为 :math:`(*, M, N)` ，其中 :math:`*` 为batch维度，其中 `*` 是batch
          维度，数据类型为int8、uint8、int16、int32、int64、float16、float32、float64。 `LU_data` 的维度必须等于或大于2。
        - **LU_pivots** (Tensor) - Tensor，打包的LU分枢轴，shape为 :math:`(*, min(M, N))` ，其中 :math:`*` 为batch维度，其中 `*` 是batch
          维度，数据类型为int8、uint8、int16、int32或int64。

    输出：
        - **pivots** (Tensor) - LU分解的置换矩阵，shape为 :math:`(*, M, M)` ，数据类型与 `LU_data` 相同。
        - **L** (Tensor) - LU分解的L矩阵，数据类型与 `LU_data` 相同。
        - **U** (Tensor) - LU分解的U矩阵，数据类型与 `LU_data` 相同。
