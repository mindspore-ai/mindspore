mindspore.ops.unique_consecutive
================================

.. py:function:: mindspore.ops.unique_consecutive(x, return_idx=False, return_counts=False, axis=None)

    返回输入张量中每个连续等效元素组中唯一的元素。

    **参数：**

    - **x** (Tensor) - 输入Tensor。
    - **return_idx** (bool, optional) - 是否返回原始输入中，各元素在返回的唯一列表中的结束位置的索引。默认值：False。
    - **return_counts** (bool, optional) - 是否返回每个唯一元素的计数。默认值：False。
    - **axis** (int, optional) - 维度。如果为None，对输入进行展平操作，返回其唯一性。如果指定，必须是int32或int64类型。默认值：None。

    **返回：**

    Tensor或包含Tensor对象的元组（ `output` 、 `idx` 、 `counts` ）。 `output` 与 `x` 具有相同的类型，用于表示唯一标量元素的输出列表。
    如果 `return_idx` 为 True，则会有一个额外的返回张量 `idx`，它的形状与 `x` 相同，表示原始输入中的元素映射到输出中的位置的索引。如果
    `return_idx` 为 True，则会有一个额外的返回张量 `counts`，表示每个唯一值或张量的出现次数。

    **异常：**

    - **TypeError** - `x` 不是Tensor。
    - **RuntimeError** – `axis` 不在 `[-ndim, ndim-1]` 范围内。
