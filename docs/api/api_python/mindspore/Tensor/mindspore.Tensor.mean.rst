mindspore.Tensor.mean
=====================

.. py:method:: mindspore.Tensor.mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

    默认情况下，移除输入所有维度，返回 `self` 中所有元素的平均值，也可仅缩小指定维度 `dim` 大小至1。 `keepdim` 控制输出和输入的维度是否相同。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **dim** (Union[int, tuple(int), list(int), Tensor]，可选) - 要减少的维度。默认值: ``None`` 。缩小所有维度，只允许常量值。假设 `self` 的秩为r，其取值范围为[-r, r)。
        - **keepdim** (bool，可选) - 如果为 ``True`` ，则保留缩小的维度并且大小为1。否则移除维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 期望返回的Tensor数据类型。默认值：``None`` 。

    返回：
        Tensor。与输入Tensor拥有相同的数据类型。

        - 如果 `dim` 设置为 ``None`` ，并且 `keepdim` 设置为 ``False`` ，则输出一个零维Tensor，表示输入Tensor中所有元素的平均值。
        - 如果 `dim` 设置为int，取值为1，并且 `keepdim` 设置为 ``False`` ，则输出的shape为：:math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `dim` 设置为tuple(int)，取值为(1, 2)，并且 `keepdim` 设置为 ``False`` ，则输出的shape为：:math:`(x_0, x_3, ..., x_R)` 。
        - 如果 `dim` 是一个一维Tensor，取值为[1, 2]，并且 `keepdim` 设置为 ``False`` ，则输出的shape为：:math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - 如果 `dim` 不是以下类型中的一种：int、tuple、list或者Tensor。
        - **TypeError** - 如果 `keepdim` 不是bool。
        - **ValueError** - 如果 `dim` 超出取值范围。

    .. py:method:: mindspore.Tensor.mean(axis=None, keep_dims=False) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.ops.mean`。
