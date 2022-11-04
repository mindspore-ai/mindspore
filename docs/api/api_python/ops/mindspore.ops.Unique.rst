mindspore.ops.Unique
====================

.. py:class:: mindspore.ops.Unique

    返回输入Tensor的唯一元素以及其对应的每个值的索引。

    输出包含Tensor `y` 和Tensor `idx` ，格式形如( `y` , `idx` )。Tensor `y` 和Tensor `idx` 的shape在大多数情况下是不同的，因为Tensor `y` 可能存在重复，并且Tensor `idx` 的shape与输入保持一致。

    要获得 `idx` 和 `y` 之间相同的shape，请参考 :class:`mindspore.ops.UniqueWithPad`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。shape为 :math:`(N,*)` ，其中 :math:`*` 表示，任意数量的附加维度。

    输出：
        Tuple，形如( `y` , `idx` )的Tensor对象， `y` 与 `input_x` 的数据类型相同，记录的是 `input_x` 中的唯一元素。 `idx` 是一个Tensor，记录的是输入 `input_x` 元素相对应的索引。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
