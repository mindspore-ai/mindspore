mindspore.ops.Shape
====================

.. py:class:: mindspore.ops.Shape

    返回输入Tensor的shape。用于静态shape。

    静态shape：无需运行图即可获得的shape。这是Tensor的固有性质，也可能是未知的。静态shape信息可以通过人工设置完成。无论图输入是什么，静态shape都不会受到影响。

    **输入：**

    - **input_x** (Tensor) - Tensor的shape是 :math:`(x_1, x_2, ..., x_R)` 。

    **输出：**

    tuple[int]，输出是由多个整数组成的tuple。 :math:`(x_1, x_2, ..., x_R)` 。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。