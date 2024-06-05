mindspore.ops.mean
==================

.. py:function:: mindspore.ops.mean(x, axis=None, keep_dims=False)

    默认情况下，移除输入所有维度，返回 `x` 中所有元素的平均值。也可仅缩小指定维度 `axis` 大小至1。 `keep_dims` 控制输出和输入的维度是否相同。

    .. note::
        Tensor类型的 `axis` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **x** (Tensor[Number]) - 输入Tensor，其数据类型为数值型。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **axis** (Union[int, tuple(int), list(int), Tensor]) - 要减少的维度。默认值:  ``None`` ，缩小所有维度。只允许常量值。假设 `x` 的秩为r，取值范围[-r,r)。
        - **keep_dims** (bool) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    返回：
        Tensor，具有与输入相同的数据类型。

        - 如果 `axis` 为 ``None`` ，且 `keep_dims` 为 ``False`` ，则输出一个零维Tensor，表示输入Tensor中所有元素的平均值。
        - 如果 `axis` 为int，取值为1，并且 `keep_dims` 为 ``False`` ，则输出的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。
        - 如果 `axis` 为一维Tensor，例如取值为[1, 2]，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、tuple、list或Tensor。
        - **TypeError** - `keep_dims` 不是bool类型。
        - **ValueError** - `axis` 超出范围。
