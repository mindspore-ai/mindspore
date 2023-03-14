mindspore.ops.shape
====================

.. py:function:: mindspore.ops.shape(input_x)

    返回输入Tensor的shape。

    参数：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1,x_2,...,x_R)` 。

    返回：
        tuple[int]，输出tuple由多个整数组成， :math:`(x_1,x_2,...,x_R)` 。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
