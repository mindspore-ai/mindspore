mindspore.ops.UniqueWithPad
===========================

.. py:class:: mindspore.ops.UniqueWithPad

    对输入一维张量中元素去重，返回一维张量中的唯一元素（使用pad_num填充）和相对索引。

    基本操作与Unique相同，但UniqueWithPad多了Pad操作。
    Unique运算符处理输入张量 `x` 后所返回的元组（ `y` ， `idx` ）， `y` 与 `idx` 的shape通常会有差别。因此，为了解决上述情况，
    UniqueWithPad操作符将用用户指定的 `pad_num` 填充 `y` 张量，使其具有与张量 `idx` 相同的形状。

    更多参考详见 :func:`mindspore.ops.unique_with_pad`。
