mindspore.ops.rsqrt
====================

.. py:function:: mindspore.ops.rsqrt(input)

    逐元素计算输入Tensor元素的平方根倒数。

    .. math::
        out_{i} =  \frac{1}{\sqrt{input_{i}}}

    参数：
        - **input** (Tensor) - rsqrt的输入Tensor，其rank需要在[0, 7]范围内且每个元素都为非负，若某个元素为负，计算结果为nan。

    返回：
        Tensor，具有与 `input` 相同的shape。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。

