mindspore.Tensor.any
====================

.. py:method:: mindspore.Tensor.any(axis=(), keep_dims=False)

    检查在指定轴方向上是否存在任意为True的Tensor元素。

    参数：
        - **axis** (Union[None, int, tuple(int)]) - 计算any的维度。当 `axis` 为None或空元组时，计算所有维度。当 `axis` 为int或tuple(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

    返回：
        Tensor。如果在指定轴方向上存在任意Tensor元素为True，则其值为True，否则其值为False。如果轴为None或空元组，则默认降维。