mindspore.ops.inner
====================

.. py:function:: mindspore.ops.inner(x, other)

    计算两个1D Tensor的点积。对于更高维度来说，计算结果为在最后一维上，逐元素乘法的和。

    .. note::
        如果 `x` 或 `other` 之一是标量，那么相当于 :code:`mindspore.ops.mul(x, other)`。

    参数：
        - **x** (Tensor) - 第一个输入。
        - **other** (Tensor) - 第二个输入。

    返回：
        Tensor，内积的结果。

    异常：
        - **ValueError** - 如果 `x` 和 `other` 都不是标量，且两者的最后一维不相同。
