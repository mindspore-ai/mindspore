mindspore.ops.unique
====================

.. py:function:: mindspore.ops.unique(input)

    对输入tensor中元素去重。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：当前默认返回 `(output, indices)`，后续默认将仅返回 `output`。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        由两个tensor组成的tuple(output, indices)。

        - **output** (Tensor) - 去重后的输出。
        - **indices** (Tensor) - 输入tensor的元素在 `output` 中的索引。
