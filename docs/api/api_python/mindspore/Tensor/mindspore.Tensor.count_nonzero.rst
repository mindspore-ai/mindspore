mindspore.Tensor.count_nonzero
==============================

.. py:method:: mindspore.Tensor.count_nonzero(dim=None) -> Tensor

    计算输入Tensor指定轴上的非零元素的数量。如果没有指定维度，则计算Tensor中所有非零元素的数量。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 `count_nonzero(axis=(), keep_dims=False, dtype=None)` 重载，仅支持 `count_nonzero(dim=None)`。

    参数：
        - **dim** (Union[None, int, tuple(int), list(int)], 可选) - 要沿其计算非零值数量的维度。默认值： ``None`` ，计算所有非零元素的个数。

    返回：
        Tensor，指定的轴上非零元素的数量。

    异常：
        - **TypeError** - `dim` 类型不是int、tuple(int)、list(int)或None。
        - **ValueError** - `dim` 取值不在 :math:`[-self.ndim, self.ndim)` 范围。

    .. py:method:: mindspore.Tensor.count_nonzero(axis=(), keep_dims=False, dtype=None) -> Tensor
        :noindex:

    计算输入Tensor指定轴上的非零元素的数量。如果没有指定维度，则计算Tensor中所有非零元素的数量。

    参数：
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 要沿其计算非零值数量的维度。默认值： ``()`` ，计算所有非零元素的个数。
        - **keep_dims** (bool, 可选) - 是否保留 `axis` 指定的维度。如果为 ``True`` ，保留对应维度size为1；如果为 ``False`` ，不保留对应维度。默认值： ``False`` 。
        - **dtype** (Union[Number, mindspore.bool]，可选) - 输出Tensor的数据类型。默认值： ``None`` 。

    返回：
        Tensor， `axis` 指定的轴上非零元素的数量。数据类型由 `dtype` 指定。

    异常：
        - **TypeError** - `axis` 不是int、tuple或者list。
        - **ValueError** - 如果 `axis` 中的任何值不在 :math:`[-self.ndim, self.ndim)` 范围内。
