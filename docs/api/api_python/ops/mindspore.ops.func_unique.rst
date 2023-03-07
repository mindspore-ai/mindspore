mindspore.ops.unique
====================

.. py:function:: mindspore.ops.unique(input)

    对输入Tensor中元素去重，并返回一个索引Tensor，包含输入Tensor中的元素在输出Tensor中的索引。

    `y` 与 `idx` 的shape通常会有差别，因为 `y` 会将元素去重，而 `idx` 的shape与 `input` 一致。
    若需要 `idx` 与 `y` 的shape相同，请参考 :class:`mindspore.ops.UniqueWithPad` 算子。

    .. warning::
        此算子为实验性算子，将来可能面临更改或删除。

    参数：
        - **input** (Tensor) - 需要被去重的Tensor。shape： :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tuple， `(y, idx)` 。 `y` 是与 `input` 数据类型相同的Tensor，包含 `input` 中去重后的元素。 `idx` 为索引Tensor，包含 `input` 中的元素在 `y` 中的索引，与 `input` 的shape相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
