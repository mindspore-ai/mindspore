mindspore.ops.shape
====================

.. py:function:: mindspore.ops.shape(input_x)

    返回输入Tensor的shape。用于静态shape。

    静态shape：不运行图就可以获得的shape。它是Tensor的固有属性，可能是未知的。静态shape信息可以通过人工设置完成。无论图的输入是什么，静态shape都不会受到影响。

    参数：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1,x_2,...,x_R)` 。

    返回：
        tuple[int]，输出tuple由多个整数组成， :math:`(x_1,x_2,...,x_R)` 。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
