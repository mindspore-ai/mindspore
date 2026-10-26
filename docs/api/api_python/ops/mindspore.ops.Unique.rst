mindspore.ops.Unique
====================

.. py:class:: mindspore.ops.Unique

    返回输入Tensor的唯一元素以及其对应的每个值的索引。

    输出包含Tensor `y` 和Tensor `idx` ，格式形如( `y` , `idx` )。Tensor `y` 和 Tensor `idx` 的 shape 在大多数情况下是不同的，因为输入中可能存在重复值，Tensor `y` 记录的是去重后的唯一元素（长度 ≤ 输入长度），而 Tensor `idx` 的 shape 与输入保持一致（长度 = 输入长度）。

    要获得 `idx` 和 `y` 之间相同的shape，请参考 :class:`mindspore.ops.UniqueWithPad`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tuple，形如( `y` , `idx` )的Tensor对象， `y` 与 `input_x` 的数据类型相同，记录的是 `input_x` 中的唯一元素。 `idx` 是一个Tensor，记录的是输入 `input_x` 中每个元素在输出 `y` 中对应的索引位置。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
