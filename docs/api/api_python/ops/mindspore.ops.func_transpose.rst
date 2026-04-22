mindspore.ops.transpose
=======================

.. py:function:: mindspore.ops.transpose(input, dims)

    根据指定的维度排列顺序对输入tensor进行维度转置。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：该接口将从当前的全排列语义调整为仅交换两个维度。

    .. note::
        GPU和CPU平台上，如果 `dims` 的元素值为负数，则其实际值为 `dims[i] + rank(input)` 。 Ascend平台不支持 `dims` 元素值为负。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dims** (Union[tuple[int], list[int]]) - 指定轴的新排列。

    返回：
        Tensor
