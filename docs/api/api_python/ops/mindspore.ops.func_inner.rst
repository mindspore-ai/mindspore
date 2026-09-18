mindspore.ops.inner
====================

.. py:function:: mindspore.ops.inner(input, other)

    返回两个一维tensor的点积。

    对于更高的维度，返回最后一个轴上逐元素乘积后的和。

    .. note::
        如果 `input` 或 `other` 之一是标量，那么 :func:`mindspore.ops.inner` 相当于 :func:`mindspore.ops.mul`。

    参数：
        - **input** (Tensor) - 第一个输入的tensor。
        - **other** (Tensor) - 第二个输入的tensor。

    返回：
        Tensor