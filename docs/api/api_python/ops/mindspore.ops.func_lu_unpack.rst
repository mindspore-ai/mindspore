mindspore.ops.lu_unpack
========================

.. py:function:: mindspore.ops.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True)

    将紧凑LU分解结果及其主元索引解包为P、L、U矩阵。

    .. note::
        - `LU_data` shape为 :math:`(*, M, N)` ， `LU_pivots` shape为 :math:`(*, \min(M, N))`， :math:`*` 表示batch数量。

    参数：
        - **LU_data** (Tensor) - 打包的LU分解数据，秩大于等于2。
        - **LU_pivots** (Tensor) - 打包的LU分解枢轴。
        - **unpack_data** (bool，可选) - 是否解压缩 `LU_data` 。如果为False，则返回的L和U为 ``None`` 。默认 ``True`` 。
        - **unpack_pivots** (bool，可选) - 是否将 `LU_pivots` 解压缩为置换矩阵P。如果为 ``False`` ，则返回的P为 ``None`` 。默认 ``True`` 。

    返回：
        由tensor组成的tuple。分别为：LU分解的置换矩阵、LU分解的L矩阵、LU分解的U矩阵。
