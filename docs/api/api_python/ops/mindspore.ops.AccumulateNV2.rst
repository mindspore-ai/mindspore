mindspore.ops.AccumulateNV2
===========================

.. py:class:: mindspore.ops.AccumulateNV2

    逐元素计算输入Tensor的累积。

    AccumulateNV2与AddN类似，但它们之间有一个显著的区别：AccumulateNV2不会等待其所有输入就绪后再求和。也就是说，不同时刻的输入会存储在内存中，AccumulateNV2能够节省内存，因为最小临时存储与输出大小成正比，而不是输入大小。

    输入：
        - **x** (Union(tuple[Tensor], list[Tensor])) - AccumulateNV2的输入，由多个Tensor组成的tuple或list，其数据类型为数值型，且每个元素的shape必须相等。

    输出：
        Tensor，数据类型和shape与输入 `x` 的每个条目相同。

    异常：
        - **TypeError** - 如果 `x` 既不是tuple也不是list。
        - **ValueError** - 如果 `x` 的元素的shape不相同。
