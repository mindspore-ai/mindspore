mindspore.Tensor.all
====================

.. py:method:: mindspore.Tensor.all(axis=(), keep_dims=False)

    检查在指定轴上所有元素是否均为True。

    参数：
        - **axis** (Union[None, int, tuple(int)]) - 计算all的维度。当 `axis` 为None或者空元组的时候，计算所有维度。当 `axis` 为int或tuple(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

    返回：
        Tensor。如果在指定轴方向上所有数组元素都为True，则其值为True，否则其值为False。如果轴为None或空元组，则默认降维。