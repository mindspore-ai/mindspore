mindspore.ops.unique_with_pad
=============================

.. py:function:: mindspore.ops.unique_with_pad(x, pad_num)

    对输入一维张量中元素去重，返回一维张量中的唯一元素（使用pad_num填充）和相对索引。

    基本操作与unique相同，但unique_with_pad多了pad操作。
    unique运算符对张量处理后所返回的元组（ `y` ， `idx` ）， `y` 与 `idx` 的shape通常会有差别。因此，为了解决上述情况，
    unique_with_pad操作符将用用户指定的 `pad_num` 填充 `y` 张量，使其具有与张量 `idx` 相同的形状。

    参数：
        - **x** (Tensor) - 需要被去重的Tensor。必须是类型为int32或int64的一维向量。
        - **pad_num** (int) - 填充值。数据类型为int32或int64。

    返回：
        Tuple， `(y, idx)` 。 `y` 是与 `x` 形状和数据类型相同的Tensor，包含 `x` 中去重后的元素，并用 `pad_num` 填充。 `idx` 为索引Tensor，包含 `x` 中的元素在 `y` 中的索引，与 `x` 的shape相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是int32也不是int64。
        - **ValueError** - `x` 不是一维张量。
