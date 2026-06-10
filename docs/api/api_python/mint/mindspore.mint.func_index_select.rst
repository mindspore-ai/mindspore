mindspore.mint.index_select
============================

.. py:function:: mindspore.mint.index_select(input, dim, index)

    返回一个新的tensor，该tensor沿维度 `dim` 按 `index` 中给定的索引对 `input` 进行选择。

    返回的tensor和输入 `input` 的维度数量相同，其第 `dim` 维度的大小和 `index` 的长度相同；其他维度和 `input` 相同。

    .. note::
        `index` 的值必须在 `[0, input.shape[dim])` 范围内；负数或超出该范围的值会导致未定义行为。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 根据索引进行选择的维度。
        - **index** (Tensor) - 包含索引的一维tensor，数据类型为int32或int64。

    返回：
        Tensor，数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 或 `index` 的类型不是Tensor。
        - **TypeError** - `dim` 的类型不是int。
        - **ValueError** - `dim` 值超出范围[-input.ndim, input.ndim - 1]。
        - **ValueError** - `index` 不是一维tensor。
