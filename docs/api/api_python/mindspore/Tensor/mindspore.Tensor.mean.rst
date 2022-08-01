mindspore.Tensor.mean
=====================

.. py:method:: mindspore.Tensor.mean(axis=(), keep_dims=False)

    返回指定维度上所有元素的均值，并降维。

    参数：
        - **axis** (Union[None, int, tuple(int), list(int)]) - 计算mean的维度。当 `axis` 为None或空元组时，计算所有维度。当 `axis` 为int、tuple(int)或list(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

    返回：
        与输入的张量具有相同的数据类型的Tensor。