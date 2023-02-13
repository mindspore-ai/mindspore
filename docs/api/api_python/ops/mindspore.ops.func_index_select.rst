mindspore.ops.index_select
==========================

.. py:function:: mindspore.ops.index_select(x, axis, index)

    返回一个新的Tensor，该Tensor沿维度 `axis` 按 `index` 中给定的顺序对 `x` 进行选择。

    返回的Tensor和输入Tensor( `x` )的维度数量相同，其第 `axis` 维度的大小和 `index` 的长度相同；其它维度和 `x` 相同。

    .. note::
        index的值必须在 `[0, x.shape[axis])` 范围内，超出该范围结果未定义。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **axis** (int) - `index` 的维度。
        - **index** (Tensor) - 包含索引的一维Tensor。

    返回：
        Tensor，数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 或 `index` 的类型不是Tensor。
        - **TypeError** - `axis` 的类型不是int。
        - **ValueError** - `axis` 值超出范围[-x.ndim, x.ndim - 1]。
        - **ValueError** - `index` 不是一维Tensor。
