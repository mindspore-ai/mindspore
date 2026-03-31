mindspore.ops.index_select
==========================

.. py:function:: mindspore.ops.index_select(input, axis, index)

    根据指定轴和索引对输入tensor进行选取，返回一个新tensor。

    .. warning::
        2.9.0（不含）之后版本中，参数 `axis` 将重命名为 `dim`。

    .. note::
        - `index` 的值必须在 `[0, input.shape[axis])` 范围内，超出该范围的结果未定义。
        - 返回的tensor和输入tensor的维度数量相同，其第 `axis` 维度的大小和 `index` 的长度相同，其他维度和 `input` 相同。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。
        - **index** (Tensor) - 指定索引，一维tensor。

    返回：
        Tensor
