mindspore.ops.accumulate_n
==========================

.. py:function:: mindspore.ops.accumulate_n(x)

    逐元素将所有输入的Tensor相加。

    accumulate_n与addn类似，但它们之间有一个显著的区别：accumulate_n不会等待其所有输入就绪后再求和。也就是说，不同时刻的输入会存储在内存中，accumulate_n能够节省内存，因为最小临时存储与输出大小成正比，而不是输入大小。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - accumulate_n的输入，由多个Tensor组成的tuple或list，其数据类型为数值型，且每个元素的shape必须相等。

    返回：
        Tensor，与 `x` 的每个Tensor具有相同的shape和数据类型。

    异常：
        - **TypeError** - `x` 既不是tuple，也不是list。
        - **ValueError** - `x` 中存在shape不同的Tensor。
