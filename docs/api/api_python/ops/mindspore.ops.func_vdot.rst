mindspore.ops.vdot
====================

.. py:function:: mindspore.ops.vdot(input, other)

    计算两个一维向量沿着维度的点积。

    计算公式如下，
    如果 `x` 是复数向量，:math:`\bar{x_{i}}` 表示向量中元素的共轭；如果 `x` 是实数向量，:math:`\bar{x_{i}}` 表示向量中元素本身。

    .. math::

        \sum_{i=1}^{n} \bar{x_{i}}{y_{i}}

    参数：
        - **input** (Tensor) - 第一个输入，必须是一维的，如果是复数，使用其共轭。
        - **other** (Tensor) - 第二个输入，必须是一维的。

    返回：
        Tensor，vdot的计算结果。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor。
        - **TypeError** - `input` 或 `other` 不是 1D Tensor。

    .. note::
        当前在GPU上不支持复数。
