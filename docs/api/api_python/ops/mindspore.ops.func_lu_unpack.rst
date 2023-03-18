mindspore.ops.lu_unpack
========================

.. py:function:: mindspore.ops.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True)

    将 `LU_data` 和 `LU_pivots` 还原为为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。通常情况下， `LU_data` 和 `LU_pivots` 是矩阵通过LU分解生成的。

    参数：
        - **LU_data** (Tensor) - 打包的LU分解数据，shape为 :math:`(*, M, N)` ，其中 :math:`*` 为batch维度，其中 `*` 是batch
          维度，数据类型为int8、uint8、int16、int32、int64、float16、float32、float64。 `LU_data` 的维度必须等于或大于2。
        - **LU_pivots** (Tensor) - 打包的LU分枢轴，shape为 :math:`(*, min(M, N))` ，其中 :math:`*` 为batch维度，其中 `*` 是batch
          维度，数据类型为int8、uint8、int16、int32或int64。
        - **unpack_data** (bool，可选) - 是否解压缩 `LU_data` 的标志。如果为False，则返回的L和U为None。默认值：True。
        - **unpack_pivots** (bool，可选) - 是否将 `LU_pivots` 解压缩为置换矩阵P的标志。如果为False，则返回的P为None。默认值：True。

    返回：
        - **pivots** (Tensor) - LU分解的置换矩阵，shape为 :math:`[*, M, M]` ，数据类型与 `LU_data` 相同。
        - **L** (Tensor) - LU分解的L矩阵，数据类型与 `LU_data` 相同。
        - **U** (Tensor) - LU分解的U矩阵，数据类型与 `LU_data` 相同。

    异常：
        - **TypeError** - 若 `LU_data` 的数据类型不是以下之一：int8、uint8、int16、int32、int64、float16、float32、float64。
        - **TypeError** - 若 `LU_pivots` 的数据类型不是以下之一：int8、uint8、int16、int32、int64。
        - **ValueError** - 若 `LU_data` 的维度小于2。
        - **ValueError** - 若 `LU_pivots` 的维度小于1。
        - **ValueError** - 若 `LU_pivots` 最后一维的大小不等于 `LU_data` 的最后两维的较小者。
        - **ValueError** - 若 `LU_data` 与 `LU_pivots` 的batch维度不匹配。
        - **ValueError** - 在CPU平台上，若 `LU_pivots` 的值不在 :math:`[1, LU\_data.shape[-2]]` 范围内。
        - **RuntimeError** - 在Ascend平台上，若 `LU_pivots` 的值不在 :math:`[1, LU\_data.shape[-2]]` 范围内。
