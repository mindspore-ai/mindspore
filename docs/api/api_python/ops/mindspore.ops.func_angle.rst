mindspore.ops.angle
===================

.. py:function:: mindspore.ops.angle(input)

    逐元素计算复数Tensor的辐角。
    输入中的元素是形式为a+bj的复数，其中a是实部，b是虚部。此函数返回的参数的形式为atan2(b，a)。

    参数：
        - **input** (Tensor) - 输入Tensor。支持类型：complex64、complex128。

    返回：
        Tensor，类型为float32或float64，shape与输入相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果输入的数据类型不是complex64或complex128。
