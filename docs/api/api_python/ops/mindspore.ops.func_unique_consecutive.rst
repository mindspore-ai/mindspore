mindspore.ops.unique_consecutive
================================

.. py:function:: mindspore.ops.unique_consecutive(input, return_idx=False, return_counts=False, axis=None)

    对输入Tensor中连续且重复的元素去重。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **return_idx** (bool, 可选) - 是否返回每个输入中元素映射到输出中位置的索引。默认值：False。
        - **return_counts** (bool, 可选) - 是否返回每个去重元素在输入所在的连续序列的计数。默认值：False。
        - **axis** (int, 可选) - 维度。如果为None，则对输入进行展平操作。如果指定，必须是int32或int64类型。默认值：None。

    返回：
        Tensor或包含Tensor对象的元组（ `output` 、 `idx` 、 `counts` ）。 

        - `output` 为去重后的输出，与 `input` 具有相同的数据类型。
        - 如果 `return_idx` 为 True，则返回张量 `idx` ，shape与 `input` 相同，表示每个输入中元素映射到输出中位置的索引。
        - 如果 `return_counts` 为 True，则返回张量 `counts` ，表示每个去重元素在输入中所在的连续序列的计数。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不支持。
        - **TypeError** - `return_idx` 不是bool。
        - **TypeError** - `return_counts` 不是bool。
        - **TypeError** - `axis` 不是int。
        - **ValueError** - `axis` 不在 `[-ndim, ndim-1]` 范围内。
