mindspore.ops.Hypot
===================

.. py:class:: mindspore.ops.Hypot

    将输入Tensor的逐个元素作为直角三角形的直角边，并计算其斜边的值。
    两个输入的shape应该是可广播的，且它们的数据类型应该是其中之一：float32、float64。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x1** (Tensor) - 第一个输入Tensor。
        - **x2** (Tensor) - 第二个输入Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中具有更高的精度的那一个。

    异常：
        - **TypeError** - 如果 `x1` 或 `x2` 的数据类型不是float32或float64。
        - **ValueError** - 如果两个输入的shape无法广播。
