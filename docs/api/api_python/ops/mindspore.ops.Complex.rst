mindspore.ops.Complex
======================

.. py:class:: mindspore.ops.Complex

    给定复数的实部与虚部，返回一个复数的Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **real** (Tensor) - 实部的取值。数据类型：float32，float64。
        - **imag** (Tensor) - 虚部的取值。数据类型：float32，float64。

    输出：
        Complex类型的Tensor。

    异常：
        - **TypeError** - 输入的数据类型不是float32或float64之一。
        - **TypeError** - 输入的实部与虚部数据类型不一致。