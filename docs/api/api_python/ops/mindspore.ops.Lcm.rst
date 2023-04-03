mindspore.ops.Lcm
=================

.. py:class:: mindspore.ops.Lcm

    逐个元素计算输入Tensor的最小公倍数。
    两个输入的shape应该是可广播的，它们的数据类型应该是其中之一：int32、int64。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x1** (Tensor) - 第一个输入Tensor。
        - **x2** (Tensor) - 第二个输入Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中具有更高的精度的那一个。

    异常：
        - **TypeError** - 如果 `x1` 或 `x2` 的数据类型不是int32或int64。
        - **ValueError** - 如果两个输入的shape无法广播。
