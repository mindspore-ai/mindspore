mindspore.Tensor.index_select
=============================

.. py:method:: mindspore.Tensor.index_select(axis, index) -> Tensor

    返回一个新的Tensor，该Tensor沿维度 `axis` 按 `index` 中给定的索引对 `self` 进行选择。
    返回的Tensor和输入Tensor( `self` )的维度数量相同，其第 `axis` 维度的大小和 `index` 的长度相同；其他维度和 `self` 相同。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 `index_select(axis, index)` 重载，仅支持 `index_select(dim, index)`。

    .. note::
        index的值必须在 `[0, self.shape[axis])` 范围内，超出该范围结果未定义。

    参数：
        - **axis** (int) - 根据索引进行选择的维度。
        - **index** (Tensor) - 包含索引的一维Tensor。

    返回：
        Tensor，数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `index` 的类型不是Tensor。
        - **TypeError** - `axis` 的类型不是int。
        - **ValueError** - `axis` 值超出范围[-input.ndim, input.ndim - 1]。
        - **ValueError** - `index` 不是一维Tensor。

    .. py:method:: mindspore.Tensor.index_select(dim, index) -> Tensor
        :noindex:

    返回一个新的Tensor，该Tensor沿维度 `dim` 按 `index` 中给定的索引对 `self` 进行选择。
    返回的Tensor和输入Tensor( `self` )的维度数量相同，其第 `dim` 维度的大小和 `index` 的长度相同；其他维度和 `self` 相同。

    .. note::
        index的值必须在 `[0, self.shape[dim])` 范围内，超出该范围结果未定义。

    参数：
        - **dim** (int) - 根据索引进行选择的维度。
        - **index** (Tensor) - 包含索引的一维Tensor。

    返回：
        Tensor，数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `index` 的类型不是Tensor。
        - **TypeError** - `dim` 的类型不是int。
        - **ValueError** - `dim` 值超出范围[-input.ndim, input.ndim - 1]。
        - **ValueError** - `index` 不是一维Tensor。
