mindspore.Tensor.amax
=====================

.. py:method:: mindspore.Tensor.amax(axis=(), keep_dims=False)

    默认情况下，使用指定维度的最大值代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 `keep_dims` 控制输出和输入的维度是否相同。

    参数：
        - **axis** (Union[None, int, tuple(int), list(int)]) - 要减少的维度。默认值: ()，缩小所有维度。只允许常量值。当 `axis` 为int、tuple(int)或list(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。
        - **keep_dims** (bool) - 如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值：False。

    返回：
        与输入的张量具有相同的数据类型的Tensor。

    异常：
        - **TypeError** - `axis` 不是以下数据类型之一：int、Tuple或List。
        - **TypeError** - `keep_dims` 不是bool类型。
        - **ValueError** - `axis` 超出范围。