mindspore.ops.inner
====================

.. py:function:: mindspore.ops.inner(input, other)

    计算两个1D Tensor的点积。

    对于1D Tensor（没有复数共轭的情况），返回两个向量的点积。

    对于更高的维度，返回最后一个轴上的和积。

    .. note::
        如果 `input` 或 `other` 之一是标量，那么 :func:`mindspore.ops.inner` 相当于 :func:`mindspore.ops.mul`。

    参数：
        - **input** (Tensor) - 第一个输入。
        - **other** (Tensor) - 第二个输入。

    返回：
        Tensor，内积的结果。

    异常：
        - **ValueError** - 如果 `input` 和 `other` 都不是标量，且两者的最后一维不相同。
