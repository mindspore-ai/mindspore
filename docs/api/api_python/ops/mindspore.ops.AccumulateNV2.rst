mindspore.ops.AccumulateNV2
===========================

.. py:class:: mindspore.ops.AccumulateNV2

    逐元素将所有输入的Tensor相加。

    有关更多详细信息，请参阅： :func:`mindspore.ops.accumulate_n` 。

    输入：
        - **x** (Union(tuple[Tensor], list[Tensor])) - AccumulateNV2的输入，由多个Tensor组成的tuple或list，其数据类型为数值型，且每个元素的shape必须相等。

    输出：
        Tensor，数据类型和shape与输入 `x` 的每个条目相同。
