mindspore.Tensor.prod
=====================

.. py:method:: mindspore.Tensor.prod(axis=(), keep_dims=False)

    默认情况下，通过将维度中的所有元素相乘来减少张量的维度。并且还可以沿轴减小"x"的维度。通过控制 `keep_dims` 判断输出和输入的维度是否相同。

    参数：
        - **axis** (Union[None, int, tuple(int), list(int)]) - 计算prod的维度。当 `axis` 为None或空元组时，计算所有维度。当 `axis` 为int、tuple(int)或list(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

    返回：
        与输入的张量具有相同的数据类型的Tensor。

    异常：
        - **TypeError** - 如果 `axis` 不是以下数据类型之一：int、tuple 或 list。
        - **TypeError** - 如果 `x` 不是Tensor类型。