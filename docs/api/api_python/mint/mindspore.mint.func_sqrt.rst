mindspore.mint.sqrt
===================

.. py:function:: mindspore.mint.sqrt(input)

    逐元素返回当前Tensor的平方根。

    .. math::
        out_{i} = \sqrt{input_{i}}

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型为number.Number。

    返回：
        Tensor，具有与 `input` 相同的shape。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。

