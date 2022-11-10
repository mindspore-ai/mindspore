mindspore.ops.lcm
==================

.. py:function:: mindspore.ops.lcm(x1, x2)

    逐元素计算两个输入Tensor的最小公倍数。两个输入Tensor的shape应该是可广播的，并且它们的数据类型应该是int32和int64其中之一。

    参数：
        - **x1** (Tensor) - 第一个输入tensor。
        - **x2** (Tensor) - 第二个输入tensor。

    返回：
        Tensor，其shape与广播后的shape相同，其数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `x1` 或 `x2` 的数据类型不是 int32 或 int64。
        - **ValueError** - `x1` 和 `x2` 的广播后的shape不相同。